# Facial Expression Recognition (FER2013)

> Đồ án môn **CS231 – Nhập môn Thị giác máy tính** — Nhóm 6
> So sánh 5 phương pháp nhận diện cảm xúc khuôn mặt trên bộ dữ liệu **FER2013** (kết hợp thêm **RAF-DB** cho phương pháp mạnh nhất), có kèm web demo real-time.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

---

## 1. Giới thiệu

**Facial Expression Recognition (FER)** là bài toán tự động nhận biết cảm xúc con người từ ảnh khuôn mặt, dựa trên Computer Vision và Deep Learning. Ứng dụng: E-learning (đánh giá sự tập trung), Driver monitoring (phát hiện buồn ngủ), Healthcare (hỗ trợ chẩn đoán tâm lý), Marketing (phân tích hành vi khách hàng)...

Dự án cài đặt và so sánh khách quan **5 phương pháp**, từ cổ điển (feature thủ công) đến hiện đại (deep learning + attention), trên cùng một pipeline dữ liệu và cùng một tập test.

### 7 lớp cảm xúc
`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

---

## 2. Cấu trúc thư mục

```
.
├── dataset.py              # Pipeline dữ liệu dùng chung: transform, DataLoader, CLAHE, RAF-DB loader
├── model.py                 # Kiến trúc ConvNeXt-Tiny + CBAM (phương pháp 5)
├── utils.py                  # Train loop, evaluate, vẽ confusion matrix / history / so sánh
│
├── method1_lbp_svm.py        # Phương pháp 1: LBP + SVM
├── method2_cnn.py            # Phương pháp 2: CNN tự xây dựng (from scratch)
├── method3_pretrained.py     # Phương pháp 3: EfficientNet-B2 fine-tune
├── method4_cbam.py           # Phương pháp 4: EfficientNet-B2 + CBAM Attention
├── train.py                  # Phương pháp 5: ConvNeXt-Tiny + CBAM (FER2013 + RAF-DB), Focal Loss, Mixup, TTA
│
├── run_inference.py          # Chạy lại inference trên test set, sinh file result_*.npy
├── check_pipeline.py         # Kiểm tra nhanh pipeline dữ liệu của cả 4 luồng tiền xử lý
├── plot_distribution.py      # Vẽ phân bố lớp trong dataset
├── plot_results.py           # Vẽ biểu đồ so sánh accuracy / F1 / confusion matrix giữa các phương pháp
│
├── server.py                 # FastAPI backend cho web demo (dự đoán real-time qua webcam/ảnh upload)
├── frontend/
│   └── index.html            # Giao diện web demo (gọi API của server.py)
│
├── saved_models/             # Checkpoint đã train sẵn + kết quả đánh giá (xem mục 6)
├── results_plots/            # Confusion matrix & biểu đồ so sánh 5 phương pháp
├── data/                     # (rỗng) — nơi đặt dataset, xem mục 4
│
├── class_distribution.png / distribution_comparison.png
├── requirements.txt
├── .gitattributes            # Cấu hình Git LFS cho các file model lớn
└── .gitignore
```

---

## 3. Cài đặt

### Yêu cầu
- Python 3.9+
- (Khuyến nghị) GPU CUDA để train — CPU vẫn chạy được cho inference / demo
- **Git LFS** — bắt buộc để tải đầy đủ các file model đã train sẵn (xem mục 6)

### Các bước

```bash
# 1. Cài Git LFS (một lần duy nhất trên máy)
git lfs install

# 2. Clone repo (LFS sẽ tự tải các file model lớn)
git clone <URL_REPO_CUA_BAN>.git
cd <ten-repo>

# 3. (Khuyến nghị) Tạo virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 4. Cài dependencies
pip install -r requirements.txt
```

---

## 4. Chuẩn bị dữ liệu

Dataset **không được commit vào repo** (dung lượng lớn + bản quyền) — bạn cần tự tải về theo cấu trúc sau:

```
data/
├── fer2013/                 # bắt buộc cho tất cả 5 phương pháp
│   ├── train/
│   │   ├── angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
│   └── test/
│       ├── angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
│
└── raf-db/                  # chỉ cần cho Phương pháp 5 (train.py)
    └── DATASET/
        ├── train/  (thư mục con 1-7, xem RAFDB_TO_FER trong dataset.py)
        └── test/
```

- **FER2013**: tải từ [Kaggle – FER2013](https://www.kaggle.com/datasets/msambare/fer2013), giải nén và sắp xếp ảnh vào đúng thư mục lớp như trên (dùng `ImageFolder` của torchvision).
- **RAF-DB**: đăng ký và tải tại [trang chủ RAF-DB](http://www.whdeng.cn/RAF/model1.html) (chỉ dùng cho `train.py`, không bắt buộc với 4 phương pháp còn lại).

Kiểm tra pipeline dữ liệu sau khi tải xong:

```bash
python check_pipeline.py
```

---

## 5. Huấn luyện

Mỗi phương pháp có script train độc lập, tự lưu checkpoint tốt nhất vào `saved_models/`.

| # | Phương pháp | Script | Lệnh chạy |
|---|---|---|---|
| 1 | LBP + SVM | `method1_lbp_svm.py` | `python method1_lbp_svm.py --data data/fer2013` |
| 2 | CNN from scratch | `method2_cnn.py` | `python method2_cnn.py --data data/fer2013 --epochs 50 --batch_size 64` |
| 3 | EfficientNet-B2 fine-tune | `method3_pretrained.py` | `python method3_pretrained.py --data data/fer2013 --epochs_s1 10 --epochs_s2 60` |
| 4 | EfficientNet-B2 + CBAM | `method4_cbam.py` | `python method4_cbam.py --data data/fer2013` |
| 5 | ConvNeXt-Tiny + CBAM (+RAF-DB) | `train.py` | `python train.py` (đường dẫn data cấu hình sẵn trong file: `FER_DIR`, `RAFDB_DIR`) |

> Phương pháp 5 dùng thêm **Focal Loss**, **Label Smoothing**, **Mixup (α=0.2)**, **Two-stage / Three-stage fine-tuning**, và **TTA x5** khi đánh giá — chi tiết trong docstring đầu file `train.py`.

Sau khi train xong, chạy lại inference trên tập test để sinh file kết quả `.npy`:

```bash
python run_inference.py --fer_dir data/fer2013
```

Vẽ biểu đồ so sánh 5 phương pháp (đọc từ các file `saved_models/result_*.npy`):

```bash
python plot_results.py
python plot_distribution.py --data data/fer2013
```

---

## 6. Kết quả (trên tập test FER2013, đã train sẵn — xem `saved_models/`)

| Phương pháp | Test Accuracy | Thời gian train (phút) | Checkpoint |
|---|---:|---:|---|
| LBP + SVM | **28.60%** | ~1.2 | `saved_models/model_lbp_svm.pkl` |
| CNN from scratch | **62.69%** | ~122 | `saved_models/model_cnn.pth` |
| EfficientNet-B2 fine-tune | **66.38%** | ~217 | `saved_models/model_pretrained.pth` |
| EfficientNet-B2 + CBAM | **67.05%** | – | `saved_models/model_cbam.pth` |
| **ConvNeXt-Tiny + CBAM** (FER2013 + RAF-DB) | **67.47%** | ~526 | `saved_models/model_convnext_cbam.pth` |

Confusion matrix và biểu đồ so sánh chi tiết theo từng lớp cảm xúc: xem thư mục [`results_plots/`](results_plots/) (đặc biệt `overview_5methods.png` và `perclass_f1_5methods.png`).

> ⚠️ **File model lớn (Git LFS):** `model_convnext_cbam.pth` và `model_convnext_cbam_scripted.pt` mỗi file ~108MB — vượt giới hạn 100MB của GitHub nên **bắt buộc dùng Git LFS** để clone đầy đủ (xem mục 3). Nếu không cần các checkpoint này, có thể bỏ qua bằng `GIT_LFS_SKIP_SMUDGE=1 git clone ...`.

---

## 7. Web demo (real-time)

Backend FastAPI expose API dự đoán cho cả 5 phương pháp cùng lúc, có face detection tự động trước khi phân loại.

```bash
python -m uvicorn server:app --reload --port 8000
```

Sau đó mở trình duyệt tại **http://localhost:8000** (server tự phục vụ luôn giao diện trong `frontend/index.html`).

### API chính
| Endpoint | Method | Mô tả |
|---|---|---|
| `/` | GET | Trang giao diện demo |
| `/health` | GET | Kiểm tra server đã load model chưa |
| `/predict/{method}` | POST | Dự đoán bằng 1 phương pháp cụ thể (`lbp`, `cnn`, `pretrained`, `cbam`, `convnext`) |
| `/predict/all` | POST | Dự đoán bằng cả 5 phương pháp cùng lúc, trả về so sánh |

---

## 8. Pipeline tiền xử lý

Hệ thống tự động chia làm 3 luồng tiền xử lý tùy kiến trúc mô hình (chi tiết trong `dataset.py::get_transforms`):

| Luồng | Kích thước | Kênh màu | Ghi chú |
|---|---|---|---|
| LBP + SVM | 48×48 | Grayscale | CLAHE làm rõ nếp nhăn biểu cảm |
| CNN | 48×48 | Grayscale | CLAHE + augmentation hình học nhẹ + Random Erasing |
| EfficientNet / ConvNeXt | 224×224 / 192×192 | RGB giả lập (nhân bản kênh xám) | Chuẩn hóa theo thống kê ImageNet |

Dữ liệu train/validation được chia bằng **Stratified Split** theo nhãn cảm xúc để giữ đúng tỉ lệ phân phối lớp giữa 2 tập.

---

## 9. Nhóm thực hiện

**Group 6 — CS231, GVHD: Mai Tiến Dũng**

| Họ Tên | MSSV | Đóng góp |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

> *(Điền lại thông tin thành viên vào bảng theo file `Work allocation table` trong slide báo cáo.)*

---

## 10. Tài liệu tham khảo

1. Deep Facial Expression Recognition: A Survey, 2020.
2. FER2013 Dataset, Kaggle.
3. Understanding Convolutional Neural Networks for NLP.
4. ImageNet Large Scale Visual Recognition Challenge.
