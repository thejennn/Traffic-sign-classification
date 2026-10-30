"""Model phân loại biển báo đã cắt (model 2).

Model định vị (model 1) là YOLO, được huấn luyện và dùng trực tiếp qua ultralytics
(xem train_detector.py và pipeline.py).
"""
from tensorflow.keras import layers, models

from data import RECOGNIZER_SIZE


def _residual_block(x, filters):
    shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    return layers.ReLU()(layers.Add()([shortcut, y]))


def build_recognizer(num_classes):
    inputs = layers.Input((RECOGNIZER_SIZE, RECOGNIZER_SIZE, 3))
    x = layers.Rescaling(1 / 255.0)(inputs)
    # Tăng cường dữ liệu chỉ chạy lúc huấn luyện
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for filters in (32, 64, 128):
        x = _residual_block(x, filters)
        x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class")(x)
    return models.Model(inputs, outputs, name="recognizer")
