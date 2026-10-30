"""Nhận diện biển báo Việt Nam trên một ảnh và lưu ảnh có khung bao.

Cách dùng:
    python predict.py duong_dan_anh.jpg
    python predict.py duong_dan_anh.jpg --out ket_qua.jpg
"""
import argparse
import os

import cv2

from pipeline import TrafficSignPipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    models = os.path.join(HERE, "models")
    pipeline = TrafficSignPipeline(
        os.path.join(models, "detector_best.pt"),
        os.path.join(models, "vn_recognizer.keras"),
        os.path.join(models, "vn_classes.txt"),
    )

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Không đọc được ảnh: {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    results = pipeline.predict(image)
    if not results:
        print("Không tìm thấy biển báo nào.")
    for r in results:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        print(f"{r['name']}  (nhận diện {r['detection_confidence']:.2f}, phân loại {r['classification_confidence']:.2f})  khung {r['box']}")

    out = args.out or os.path.splitext(args.image)[0] + "_ket_qua.jpg"
    cv2.imwrite(out, bgr)
    print(f"Đã lưu ảnh kết quả: {out}")


if __name__ == "__main__":
    main()
