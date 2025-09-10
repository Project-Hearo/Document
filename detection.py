#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import logging
import time

# 80개의 COCO 클래스 이름을 코드에 직접 내장
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebcamDetector:
    """
    웹캠 영상을 받아 TFLite 모델로 객체를 탐지하는 클래스
    """
    def __init__(self, model_path, camera_index, confidence_threshold, nms_threshold):
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # TFLite 모델 로드 및 인터프리터 생성
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 모델의 입력 및 출력 정보 가져오기
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 입력 크기 정보 (예: [1, 640, 640, 3])
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # 클래스별로 다른 색상의 바운딩 박스를 그리기 위한 색상맵 생성
        self.colors = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"오류: 카메라를 열 수 없습니다 (인덱스: {self.camera_index})")
            sys.exit(1)

    def preprocess_image(self, image):
        """입력 이미지를 모델에 맞게 전처리"""
        resized_image = cv2.resize(image, (self.input_width, self.input_height))
        # BGR -> RGB 변환, 차원 확장(batch), float32 변환
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_image, axis=0).astype(np.float32)
        return input_data, resized_image

    def postprocess_output(self, output_data, original_shape, resized_shape):
        """모델의 출력값을 후처리하여 바운딩 박스, 신뢰도, 클래스 ID 추출"""
        detections = output_data[0]
        
        conf_mask = detections[:, 4] > self.confidence_threshold
        detections = detections[conf_mask]
        
        if len(detections) == 0:
            return [], [], []

        class_scores = detections[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = detections[:, 4] * class_scores[np.arange(len(detections)), class_ids]
        
        score_mask = scores > self.confidence_threshold
        detections = detections[score_mask]
        scores = scores[score_mask]
        class_ids = class_ids[score_mask]

        # ==========================================================
        # ## 수정된 부분: 'person' 클래스(ID: 0)만 필터링 ##
        person_mask = class_ids == 0
        detections = detections[person_mask]
        scores = scores[person_mask]
        class_ids = class_ids[person_mask]
        # ==========================================================
        
        if len(detections) == 0:
            return [], [], []

        boxes_cxcywh = detections[:, :4]
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes = np.stack([y1, x1, y2, x2], axis=1)

        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=50, iou_threshold=self.nms_threshold
        )
        
        final_boxes, final_scores, final_class_ids = [], [], []

        h_orig, w_orig = original_shape
        h_res, w_res = resized_shape
        
        for index in selected_indices:
            box = boxes[index]
            y1, x1, y2, x2 = box
            
            x1_orig = int(x1 * (w_orig / w_res))
            y1_orig = int(y1 * (h_orig / h_res))
            x2_orig = int(x2 * (w_orig / w_res))
            y2_orig = int(y2 * (h_orig / h_res))
            
            final_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
            final_scores.append(scores[index])
            final_class_ids.append(class_ids[index])

        return final_boxes, final_scores, final_class_ids

    def draw_detections(self, frame, boxes, scores, class_ids):
        """탐지된 객체들을 프레임에 그림"""
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            
            label = COCO_CLASSES[class_id]
            color = self.colors[class_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f'{label}: {score:.2f}'
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def run(self):
        """메인 실행 루프"""
        logger.info("실시간 '사람' 탐지를 시작합니다. 종료하려면 'q'를 누르세요.")
        
        prev_time = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("오류: 프레임을 읽어올 수 없습니다.")
                break

            input_data, resized_frame = self.preprocess_image(frame)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            boxes, scores, class_ids = self.postprocess_output(
                output_data, frame.shape[:2], resized_frame.shape[:2]
            )

            self.draw_detections(frame, boxes, scores, class_ids)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Person Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                logger.info(f"신뢰도 임계값 변경: {self.confidence_threshold:.1f}")
            elif key == ord('v'):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.1)
                logger.info(f"신뢰도 임계값 변경: {self.confidence_threshold:.1f}")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """명령행 인수 파싱 및 메인 로직 실행"""
    parser = argparse.ArgumentParser(
        description="Yolo-X TFLite 모델을 사용한 실시간 '사람' 탐지",
        formatter_class=arg.RawDescriptionHelpFormatter,
        epilog="""
키보드 단축키:
  ESC 또는 'q': 프로그램 종료
  'c': 신뢰도 임계값 감소 (0.1씩)
  'v': 신뢰도 임계값 증가 (0.1씩)
        """
    )
    parser.add_argument('--model', type=str, default='Yolo-X_w8a8.tflite', help='TFLite 모델 파일 경로')
    parser.add_argument('--camera', type=int, default=0, help='웹캠 인덱스')
    parser.add_argument('--confidence', type=float, default=0.5, help='신뢰도 임계값')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS 임계값')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"모델 파일을 찾을 수 없습니다: {args.model}")
        sys.exit(1)

    try:
        detector = WebcamDetector(
            model_path=args.model,
            camera_index=args.camera,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms
        )
        detector.run()
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
