"""Ghép hai model để nhận diện biển báo Việt Nam trên một ảnh bất kỳ.

Luồng xử lý:
    ảnh -> YOLO (model 1, tìm khung bao) -> cắt vùng biển báo -> model phân loại (model 2)
"""
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from data import RECOGNIZER_SIZE, crop_with_margin


class TrafficSignPipeline:
    def __init__(self, detector_path, recognizer_path, classes_path, conf=0.25):
        self.detector = YOLO(detector_path)
        self.recognizer = tf.keras.models.load_model(recognizer_path)
        with open(classes_path, encoding="utf-8") as f:
            self.class_names = f.read().splitlines()
        self.conf = conf

    def detect(self, image):
        """Model 1: trả về danh sách khung bao (x1, y1, x2, y2, độ tin cậy) theo pixel."""
        # YOLO nhận ảnh BGR (quy ước của OpenCV), còn ảnh đầu vào ở đây là RGB
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = self.detector.predict(bgr, conf=self.conf, verbose=False)[0]
        return [(*box.xyxy[0].tolist(), float(box.conf[0])) for box in result.boxes]

    def classify(self, crop):
        """Model 2: trả về (tên lớp, độ tin cậy) của một ảnh biển báo đã cắt."""
        crop = cv2.resize(crop, (RECOGNIZER_SIZE, RECOGNIZER_SIZE))
        probs = self.recognizer.predict(crop[None], verbose=0)[0]
        class_id = int(probs.argmax())
        return self.class_names[class_id], float(probs[class_id])

    def predict(self, image):
        """Nhận diện mọi biển báo trong ảnh RGB. Trả về danh sách kết quả."""
        detections = [d for d in self.detect(image) if d[2] - d[0] > 0 and d[3] - d[1] > 0]
        crops = [
            cv2.resize(crop_with_margin(image, d[:4]), (RECOGNIZER_SIZE, RECOGNIZER_SIZE))
            for d in detections
        ]
        if not crops:
            return []
        # Gộp mọi vùng cắt của khung hình vào một lần suy luận
        probs = self.recognizer(np.stack(crops), training=False).numpy()
        results = []
        for (x1, y1, x2, y2, det_conf), p in zip(detections, probs):
            class_id = int(p.argmax())
            results.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "name": self.class_names[class_id],
                "detection_confidence": det_conf,
                "classification_confidence": float(p[class_id]),
            })
        return results
