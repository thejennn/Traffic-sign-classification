"""Ghép hai model để nhận diện biển báo trên một ảnh bất kỳ.

Luồng xử lý:
    ảnh -> model định vị (khung bao) -> cắt vùng biển báo -> model phân loại -> tên biển báo
"""
import cv2
import numpy as np
import tensorflow as tf

from data import LOCALIZER_SIZE, RECOGNIZER_SIZE, crop_with_margin

SIGN_NAMES = [
    "Giới hạn tốc độ 20 km/h",
    "Giới hạn tốc độ 30 km/h",
    "Giới hạn tốc độ 50 km/h",
    "Giới hạn tốc độ 60 km/h",
    "Giới hạn tốc độ 70 km/h",
    "Giới hạn tốc độ 80 km/h",
    "Hết giới hạn tốc độ 80 km/h",
    "Giới hạn tốc độ 100 km/h",
    "Giới hạn tốc độ 120 km/h",
    "Cấm vượt",
    "Cấm vượt đối với xe trên 3,5 tấn",
    "Giao nhau với đường ưu tiên",
    "Đường ưu tiên",
    "Nhường đường",
    "Dừng lại",
    "Cấm xe",
    "Cấm xe trên 3,5 tấn",
    "Cấm đi vào",
    "Chú ý chung",
    "Đường cong nguy hiểm bên trái",
    "Đường cong nguy hiểm bên phải",
    "Đường cong liên tiếp",
    "Đường gồ ghề",
    "Đường trơn",
    "Đường hẹp bên phải",
    "Công trường",
    "Tín hiệu đèn giao thông",
    "Người đi bộ",
    "Trẻ em qua đường",
    "Xe đạp qua đường",
    "Cảnh báo băng tuyết",
    "Động vật hoang dã qua đường",
    "Hết mọi giới hạn tốc độ và cấm vượt",
    "Chỉ được rẽ phải phía trước",
    "Chỉ được rẽ trái phía trước",
    "Chỉ được đi thẳng",
    "Đi thẳng hoặc rẽ phải",
    "Đi thẳng hoặc rẽ trái",
    "Đi sát bên phải",
    "Đi sát bên trái",
    "Vòng xuyến",
    "Hết cấm vượt",
    "Hết cấm vượt đối với xe trên 3,5 tấn",
]


class TrafficSignPipeline:
    def __init__(self, localizer_path, recognizer_path):
        self.localizer = tf.keras.models.load_model(localizer_path)
        self.recognizer = tf.keras.models.load_model(recognizer_path)

    def locate(self, image):
        """Model 1: trả về khung bao (x1, y1, x2, y2) theo pixel của ảnh gốc."""
        h, w = image.shape[:2]
        small = cv2.resize(image, (LOCALIZER_SIZE, LOCALIZER_SIZE))
        box = self.localizer.predict(small[None], verbose=0)[0]
        x1, y1, x2, y2 = box * np.array([w, h, w, h])
        return int(x1), int(y1), int(x2), int(y2)

    def classify(self, crop):
        """Model 2: trả về (mã lớp, độ tin cậy) của một ảnh biển báo đã cắt."""
        crop = cv2.resize(crop, (RECOGNIZER_SIZE, RECOGNIZER_SIZE))
        probs = self.recognizer.predict(crop[None], verbose=0)[0]
        class_id = int(np.argmax(probs))
        return class_id, float(probs[class_id])

    def predict(self, image):
        """Nhận diện biển báo trong ảnh RGB. Trả về None nếu không tìm thấy vùng hợp lệ."""
        box = self.locate(image)
        crop = crop_with_margin(image, box)  # cùng viền 15% như lúc huấn luyện
        if crop.size == 0:
            return None
        class_id, confidence = self.classify(crop)
        return {
            "box": box,
            "class_id": class_id,
            "name": SIGN_NAMES[class_id],
            "confidence": confidence,
        }
