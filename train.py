"""Huấn luyện cả hai model: định vị (model 1) và phân loại (model 2).

Cách dùng:
    python train.py --data path/to/gtsrb --out models
"""
import argparse
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data import load_localizer_data, load_recognizer_data, read_csv
from models import build_localizer, build_recognizer


def callbacks(name, out_dir):
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, f"{name}.keras"), save_best_only=True),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df = read_csv(args.data, "Train.csv")
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=43, stratify=train_df["ClassId"])

    # Model 1: định vị
    x_train, y_box_train = load_localizer_data(args.data, train_df)
    x_val, y_box_val = load_localizer_data(args.data, val_df)
    localizer = build_localizer()
    localizer.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    localizer.fit(
        x_train, y_box_train, validation_data=(x_val, y_box_val),
        epochs=args.epochs, batch_size=64, callbacks=callbacks("localizer", args.out),
    )

    # Model 2: phân loại
    x_train, y_train = load_recognizer_data(args.data, train_df)
    x_val, y_val = load_recognizer_data(args.data, val_df)
    recognizer = build_recognizer()
    recognizer.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    recognizer.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=64, callbacks=callbacks("recognizer", args.out),
    )


if __name__ == "__main__":
    main()
