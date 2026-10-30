"""Đọc dữ liệu GTSRB: ảnh, khung bao (ROI) và nhãn lớp.

Cấu trúc thư mục dữ liệu (tải từ Kaggle):
    <root>/Train.csv
    <root>/Test.csv
    <root>/Train/<class>/*.png
"""
import os

import cv2
import numpy as np
import pandas as pd

NUM_CLASSES = 43
LOCALIZER_SIZE = 128  # ảnh đầu vào của model định vị
RECOGNIZER_SIZE = 48  # ảnh biển báo đã cắt, đầu vào của model phân loại


def read_csv(root, name):
    return pd.read_csv(os.path.join(root, name))


def read_image(root, rel_path):
    image = cv2.imread(os.path.join(root, rel_path))
    if image is None:
        raise FileNotFoundError(os.path.join(root, rel_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalized_box(row):
    """Khung bao đưa về [0, 1] theo kích thước ảnh gốc."""
    w, h = row["Width"], row["Height"]
    return np.array(
        [row["Roi.X1"] / w, row["Roi.Y1"] / h, row["Roi.X2"] / w, row["Roi.Y2"] / h],
        dtype=np.float32,
    )


def crop_with_margin(image, box, margin=0.15):
    """Cắt vùng biển báo, nới thêm một chút viền để có ngữ cảnh."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - margin * bw))
    y1 = int(max(0, y1 - margin * bh))
    x2 = int(min(w, x2 + margin * bw))
    y2 = int(min(h, y2 + margin * bh))
    return image[y1:y2, x1:x2]


def load_localizer_data(root, df):
    """Trả về (ảnh 128x128, khung bao chuẩn hoá) cho model định vị."""
    images, boxes = [], []
    for _, row in df.iterrows():
        image = read_image(root, row["Path"])
        images.append(cv2.resize(image, (LOCALIZER_SIZE, LOCALIZER_SIZE)))
        boxes.append(normalized_box(row))
    return np.array(images, dtype=np.uint8), np.array(boxes, dtype=np.float32)


def load_recognizer_data(root, df):
    """Trả về (ảnh biển báo đã cắt 48x48, nhãn lớp) cho model phân loại."""
    crops, labels = [], []
    for _, row in df.iterrows():
        image = read_image(root, row["Path"])
        box = (row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"])
        crop = crop_with_margin(image, box)
        crops.append(cv2.resize(crop, (RECOGNIZER_SIZE, RECOGNIZER_SIZE)))
        labels.append(row["ClassId"])
    return np.array(crops, dtype=np.uint8), np.array(labels, dtype=np.int64)
