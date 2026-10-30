"""Đánh giá hệ thống trên tập kiểm tra.

- IoU của model định vị (khung bao dự đoán so với khung bao thật).
- Độ chính xác từng model và độ chính xác cuối cùng của cả pipeline.

Cách dùng:
    python evaluate.py --data path/to/gtsrb --models models
"""
import argparse
import os

import numpy as np
from tqdm import tqdm

from data import crop_with_margin, normalized_box, read_csv, read_image
from pipeline import TrafficSignPipeline


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    pipeline = TrafficSignPipeline(
        os.path.join(args.models, "localizer.keras"),
        os.path.join(args.models, "recognizer.keras"),
    )
    df = read_csv(args.data, "Test.csv")

    ious, loc_hits, cls_hits, e2e_hits = [], [], [], 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = read_image(args.data, row["Path"])
        h, w = image.shape[:2]
        true_box = normalized_box(row) * np.array([w, h, w, h])

        result = pipeline.predict(image)
        if result is None:
            ious.append(0.0)
            continue

        score = iou(result["box"], true_box)
        ious.append(score)
        loc_hits.append(score >= 0.5)

        # Phân loại trên đúng vùng khung thật để tách riêng lỗi của model 2
        crop = crop_with_margin(image, tuple(true_box))
        predicted_class, _ = pipeline.classify(crop)
        cls_hits.append(predicted_class == row["ClassId"])

        e2e_hits += int(result["class_id"] == row["ClassId"] and score >= 0.5)

    n = len(df)
    print(f"Mean IoU localizer:            {np.mean(ious):.4f}")
    print(f"Localizer IoU >= 0.5:          {np.mean(loc_hits):.4f}")
    print(f"Recognizer accuracy (true ROI): {np.mean(cls_hits):.4f}")
    print(f"End-to-end accuracy:           {e2e_hits / n:.4f}")


if __name__ == "__main__":
    main()
