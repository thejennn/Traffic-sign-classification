"""Huấn luyện model phân loại biển báo (model 2) theo hai bước.

1. Pretrain trên GTSRB (43 lớp, tải từ Kaggle) để học đặc trưng biển báo.
2. Tinh chỉnh trên biển báo Việt Nam (6 lớp, bộ Roboflow), với các trọng số
   của bước 1 được giữ lại cho mọi lớp trừ lớp đầu ra.

Cách dùng:
    python train_recognizer.py --gtsrb path/to/gtsrb --vn path/to/rf-vn --out models
"""
import argparse
import csv
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data import NUM_CLASSES, crop_with_margin, load_recognizer_data, read_csv, read_image, RECOGNIZER_SIZE
from models import build_recognizer

VN_SPLITS = {"train": "train", "val": "valid", "test": "test"}


def vn_class_names(src):
    """Danh sách lớp Việt Nam, sắp xếp giống prepare_vn_dataset.py để nhãn khớp nhau."""
    names = set()
    for split in VN_SPLITS.values():
        with open(os.path.join(src, split, "_annotations.csv"), encoding="utf-8") as f:
            names.update(row["class"] for row in csv.DictReader(f) if row["class"])
    return sorted(names)


def load_vn_crops(src, split, class_index):
    """Cắt biển báo Việt Nam theo khung bao thật của Roboflow."""
    split_dir = os.path.join(src, VN_SPLITS[split])
    crops, labels = [], []
    with open(os.path.join(split_dir, "_annotations.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["class"]:
                continue
            image = read_image(split_dir, row["filename"])
            box = tuple(float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            crop = crop_with_margin(image, box)
            crops.append(cv2.resize(crop, (RECOGNIZER_SIZE, RECOGNIZER_SIZE)))
            labels.append(class_index[row["class"]])
    return np.array(crops, dtype=np.uint8), np.array(labels, dtype=np.int64)


def callbacks(path):
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True),
    ]


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtsrb", required=True)
    parser.add_argument("--vn", required=True)
    parser.add_argument("--out", default="models")
    parser.add_argument("--pre-epochs", type=int, default=15)
    parser.add_argument("--ft-epochs", type=int, default=30)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Bước 1: pretrain trên GTSRB
    gt = read_csv(args.gtsrb, "Train.csv")
    gt_train, gt_val = train_test_split(gt, test_size=0.2, random_state=43, stratify=gt["ClassId"])
    x_tr, y_tr = load_recognizer_data(args.gtsrb, gt_train)
    x_va, y_va = load_recognizer_data(args.gtsrb, gt_val)
    pretrained = build_recognizer(NUM_CLASSES)
    compile_model(pretrained, 1e-3)
    pretrained.fit(
        x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.pre_epochs, batch_size=64,
        callbacks=callbacks(os.path.join(args.out, "gtsrb_pretrained.keras")),
    )

    # Bước 2: tinh chỉnh trên biển báo Việt Nam
    names = vn_class_names(args.vn)
    class_index = {n: i for i, n in enumerate(names)}
    x_tr, y_tr = load_vn_crops(args.vn, "train", class_index)
    x_va, y_va = load_vn_crops(args.vn, "val", class_index)
    x_te, y_te = load_vn_crops(args.vn, "test", class_index)

    recognizer = build_recognizer(len(names))
    for old_layer, new_layer in zip(pretrained.layers[:-1], recognizer.layers[:-1]):
        new_layer.set_weights(old_layer.get_weights())

    # Lớp hiếm được trọng số cao hơn để không bị bỏ qua
    counts = np.bincount(y_tr, minlength=len(names))
    class_weight = {i: float(counts.sum() / (len(names) * c)) for i, c in enumerate(counts) if c > 0}

    compile_model(recognizer, 3e-4)
    recognizer.fit(
        x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.ft_epochs, batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks(os.path.join(args.out, "vn_recognizer.keras")),
    )

    loss, acc = recognizer.evaluate(x_te, y_te, verbose=0)
    print(f"Tập test biển báo Việt Nam: độ chính xác = {acc:.4f}, loss = {loss:.4f}")
    with open(os.path.join(args.out, "vn_classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))


if __name__ == "__main__":
    main()
