"""Nhận diện biển báo Việt Nam theo thời gian thực từ webcam.

Cách dùng:
    python live.py              # dùng webcam mặc định (số 0)
    python live.py --camera 1   # chọn webcam khác

Nhấn q để thoát.
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import TrafficSignPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATHS = [r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def load_font(size=22):
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_results(frame_bgr, results, font):
    """Vẽ khung và tên biển báo (có dấu tiếng Việt) lên khung hình."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for r in results:
        x1, y1, x2, y2 = r["box"]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        label = f"{r['name']} ({r['classification_confidence']:.0%})"
        text_y = max(0, y1 - 28)
        left, top, right, bottom = draw.textbbox((x1, text_y), label, font=font)
        draw.rectangle((left, top, right + 6, bottom + 4), fill=(0, 200, 0))
        draw.text((x1 + 3, text_y), label, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    models = os.path.join(HERE, "models")
    pipeline = TrafficSignPipeline(
        os.path.join(models, "detector_best.pt"),
        os.path.join(models, "vn_recognizer.keras"),
        os.path.join(models, "vn_classes.txt"),
    )
    font = load_font()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam số {args.camera}. Thử --camera 1 hoặc kiểm tra quyền camera.")

    print("Đang chạy. Nhấn q để thoát.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Không đọc được khung hình từ webcam.")
            break
        results = pipeline.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow("Nhan dien bien bao", draw_results(frame, results, font))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
