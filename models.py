"""Hai model tách riêng:

- build_localizer(): model 1, tìm khung bao của biển báo trên ảnh đầy đủ.
- build_recognizer(): model 2, phân loại biển báo đã cắt thành 43 lớp.
"""
from tensorflow.keras import layers, models

from data import LOCALIZER_SIZE, NUM_CLASSES, RECOGNIZER_SIZE


def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.MaxPool2D()(x)


def build_localizer():
    inputs = layers.Input((LOCALIZER_SIZE, LOCALIZER_SIZE, 3))
    x = layers.Rescaling(1 / 255.0)(inputs)
    x = layers.RandomTranslation(0.05, 0.05)(x)  # chỉ hoạt động lúc huấn luyện
    for filters in (32, 64, 128, 128, 256):
        x = _conv_block(x, filters)
    x = layers.Flatten()(x)  # giữ vị trí không gian, quan trọng cho bài toán khung bao
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    boxes = layers.Dense(4, activation="sigmoid", name="box")(x)
    return models.Model(inputs, boxes, name="localizer")


def _residual_block(x, filters):
    shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    return layers.ReLU()(layers.Add()([shortcut, y]))


def build_recognizer():
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
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="class")(x)
    return models.Model(inputs, outputs, name="recognizer")
