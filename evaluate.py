"""Đánh giá pipeline trên tập test của bộ biển báo Việt Nam.

- Phát hiện: tỉ lệ biển báo thật được tìm thấy (IoU >= 0.5 với một khung dự đoán).
- Phân loại: độ chính xác của model 2 trên đúng vùng biển báo thật.
- Cả hệ thống: biển báo được tìm thấy VÀ phân loại đúng.

Cách dùng:
    python evaluate.py --vn path/to/rf-vn --models models
"""
import argparse
import csv
import os

from tqdm import tqdm

from data import crop_with_margin, read_image
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
    parser.add_argument("--vn", required=True)
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    pipeline = TrafficSignPipeline(
        os.path.join(args.models, "detector_best.pt"),
        os.path.join(args.models, "vn_recognizer.keras"),
        os.path.join(args.models, "vn_classes.txt"),
    )

    test_dir = os.path.join(args.vn, "test")
    with open(os.path.join(test_dir, "_annotations.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["class"]]

    # Gom các biển báo thật theo ảnh để chạy mỗi ảnh một lần
    by_image = {}
    for row in rows:
        by_image.setdefault(row["filename"], []).append(row)

    total = found = cls_correct = e2e_correct = 0
    for filename, gts in tqdm(by_image.items()):
        image = read_image(test_dir, filename)
        preds = pipeline.predict(image)
        for gt in gts:
            total += 1
            gt_box = tuple(float(gt[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            # Phân loại đúng vùng thật để tách lỗi của model 2
            crop = crop_with_margin(image, gt_box)
            cls_name, _ = pipeline.classify(crop)
            cls_correct += cls_name == gt["class"]

            matches = [p for p in preds if iou(p["box"], gt_box) >= 0.5]
            if matches:
                found += 1
                best = max(matches, key=lambda p: p["detection_confidence"])
                e2e_correct += best["name"] == gt["class"]

    print(f"Số biển báo thật trên tập test: {total}")
    print(f"Phát hiện được (IoU >= 0.5):      {found / total:.4f}")
    print(f"Phân loại đúng (vùng thật):       {cls_correct / total:.4f}")
    print(f"Cả hệ thống đúng:                 {e2e_correct / total:.4f}")


if __name__ == "__main__":
    main()
