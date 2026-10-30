"""Chuyển bộ Roboflow "Vietnamese Traffic Signs Detection" sang định dạng YOLO.

Bộ gốc có sẵn các split train/valid/test, mỗi split một file _annotations.csv
với cột: filename,width,height,class,xmin,ymin,xmax,ymax.

Cách dùng:
    python prepare_vn_dataset.py --src path/to/rf-vn --out path/to/vn-yolo
"""
import argparse
import csv
import os
import shutil

import yaml

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def to_yolo(xmin, ymin, xmax, ymax, w, h):
    xc = (xmin + xmax) / 2 / w
    yc = (ymin + ymax) / 2 / h
    return xc, yc, (xmax - xmin) / w, (ymax - ymin) / h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Lấy danh sách lớp từ cả ba split để thứ tự nhất quán
    class_names = set()
    for src_split in SPLITS.values():
        with open(os.path.join(args.src, src_split, "_annotations.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["class"]:
                    class_names.add(row["class"])
    class_names = sorted(class_names)
    class_index = {name: i for i, name in enumerate(class_names)}

    for out_split, src_split in SPLITS.items():
        img_dir = os.path.join(args.out, "images", out_split)
        lbl_dir = os.path.join(args.out, "labels", out_split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        src_dir = os.path.join(args.src, src_split)
        labels = {}
        with open(os.path.join(src_dir, "_annotations.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                shutil.copy(os.path.join(src_dir, row["filename"]), os.path.join(img_dir, row["filename"]))
                lines = labels.setdefault(row["filename"], [])
                if not row["class"]:  # ảnh không có biển báo: nhãn rỗng
                    continue
                w, h = float(row["width"]), float(row["height"])
                box = to_yolo(float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]), w, h)
                lines.append(f"{class_index[row['class']]} " + " ".join(f"{v:.6f}" for v in box))

        for filename, lines in labels.items():
            stem = os.path.splitext(filename)[0]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        print(f"{out_split}: {len(labels)} ảnh")

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": os.path.abspath(args.out),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: n for i, n in enumerate(class_names)},
            },
            f,
            allow_unicode=True,
        )
    print("lớp:", class_names)


if __name__ == "__main__":
    main()
