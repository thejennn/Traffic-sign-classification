# Nhận diện biển báo giao thông Việt Nam (YOLO + CNN)

Hệ thống nhận diện biển báo giao thông Việt Nam theo hai bước:

1. **Model 1 – định vị (YOLOv8n):** tìm khung bao của biển báo trong ảnh hoặc khung hình webcam.
2. **Model 2 – phân loại (CNN):** phân loại vùng biển báo đã cắt thành một trong **6 lớp**.

Model 2 được pretrain trên GTSRB (43 lớp biển báo Đức, tải từ Kaggle) để học đặc trưng biển báo, rồi tinh chỉnh trên bộ biển báo Việt Nam của Roboflow.

> **Phạm vi hiện tại:** hệ thống chỉ nhận được **6 loại biển** sau đây. Biển khác sẽ không được nhận đúng.
> - Cấm quay đầu (No U-turn)
> - Cấm đi ngược chiều (No going opposite)
> - Cấm dừng (No stopping)
> - Cấm đỗ (No waiting)
> - Vòng xuyến phía trước (Roundabout ahead)
> - Giới hạn tốc độ 60 km/h (Speed limit 60kmh)

## Kết quả

Đánh giá trên tập test của bộ Roboflow (140 biển báo thật trên 103 ảnh):

| Chỉ số | Kết quả |
|---|---|
| Phát hiện được (IoU ≥ 0,5) | 94,3% |
| Phân loại đúng (trên vùng biển thật) | 98,6% |
| Cả hệ thống đúng | 93,6% |

Model YOLO đạt mAP50 = 0,947 trên tập validation. Tập test chỉ có 140 biển báo nên sai số khá lớn.

**Hạn chế đã biết:**
- Dữ liệu chụp từ xe, độ phân giải 1920×1080. Ảnh từ webcam hoặc màn hình có thể cho kết quả kém hơn.
- Biển phải đủ lớn trong khung hình. Biển nhỏ hoặc xa thường không được phát hiện.
- Chưa đo độ chính xác trên tập test GTSRB chính thức.

## Cài đặt

```bash
pip install -r requirements.txt
```

Nếu muốn YOLO chạy trên GPU NVIDIA, cài bản PyTorch có CUDA sau khi cài file trên:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
```

## Chạy nhận diện

Trên một ảnh:

```bash
python predict.py duong_dan_anh.jpg
```

Trên webcam (nhấn `q` để thoát):

```bash
python live.py
python live.py --camera 1   # nếu máy có nhiều camera
```

## Huấn luyện lại

1. Chuẩn bị bộ Roboflow đã giải nén, chuyển sang định dạng YOLO:
   ```bash
   python prepare_vn_dataset.py --src path/to/rf-vn --out path/to/vn-yolo
   ```
2. Huấn luyện model định vị:
   ```bash
   yolo detect train model=yolov8n.pt data=path/to/vn-yolo/data.yaml epochs=30 imgsz=640 batch=8 device=0
   ```
3. Huấn luyện model phân loại (cần bộ GTSRB và bộ Roboflow):
   ```bash
   python train_recognizer.py --gtsrb path/to/gtsrb --vn path/to/rf-vn --out models
   ```
4. Đánh giá trên tập test:
   ```bash
   python evaluate.py --vn path/to/rf-vn --models models
   ```

Sau khi huấn luyện, copy `best.pt` của YOLO thành `models/detector_best.pt`. Các file model đã huấn luyện được đặt sẵn trong thư mục `models/`.

## Dữ liệu

- **Biển báo Việt Nam:** [Vietnamese Traffic Signs Detection (Roboflow Universe)](https://universe.roboflow.com/vietnamese-traffic-signs-detection-jbfbz/vietnamese-traffic-signs-detection), giấy phép CC BY 4.0.
- **GTSRB:** [German Traffic Sign Recognition Benchmark (Kaggle)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), dùng để pretrain model phân loại.

## Cấu trúc

| File | Vai trò |
|---|---|
| `predict.py` | Nhận diện một ảnh, lưu ảnh có khung bao |
| `live.py` | Nhận diện thời gian thực từ webcam |
| `pipeline.py` | Ghép YOLO và model phân loại |
| `models.py` | Kiến trúc model phân loại |
| `data.py` | Đọc dữ liệu GTSRB và cắt vùng biển báo |
| `prepare_vn_dataset.py` | Chuyển bộ Roboflow sang định dạng YOLO |
| `train_recognizer.py` | Pretrain trên GTSRB, tinh chỉnh trên biển báo Việt Nam |
| `evaluate.py` | Đánh giá hệ thống trên tập test |
| `models/` | Model đã huấn luyện và danh sách lớp |

## Giấy phép

MIT. Bản quyền thuộc về thejennn.
