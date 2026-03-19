# 🎙️ L2-ARCTIC Phoneme-Level Mispronunciation Detection

Hệ thống AI chuyên biệt nhằm Đánh giá và Phát hiện Lỗi phát âm Tiếng Anh ở **cấp độ Âm vị (Phoneme-level)**. Dự án sử dụng mô hình học sâu **Wav2Vec2-Base-960h** kết hợp với thuật toán gióng hàng **Levenshtein Distance** và tự điển **G2P**, giúp phát hiện chính xác 3 loại lỗi phát âm thường gặp của người học tiếng Anh:
* ❌ **Substitution:** Đọc sai âm (VD: Phát âm `TH` thành `S` hoặc `D`).
* ⚠️ **Deletion:** Nuốt âm (VD: Quên đọc âm đuôi `S`, `T`, `L`).
* 🔴 **Insertion:** Đọc thừa âm (Chèn thêm âm rác vào từ).

Hệ thống được thiết kế theo kiến trúc chuẩn MLOps, tinh chỉnh (Fine-tuning) từ bộ tệp dữ liệu âm thanh người học tiếng Anh [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/).

---

## 📂 Kiến trúc Mã nguồn (Repository Structure)

Dự án được phân chia module rõ ràng để tái sử dụng và dễ dàng bảo trì:

```text
├── phoneme_assessment/          # 🧠 Package Lõi (Core Backend)
│   ├── dataset.py               # Xử lý Pytorch Dataset & Padding (DataCollator CTC)
│   ├── model.py                 # Khởi tạo và Đóng băng (Freeze) mô hình Wav2Vec2
│   ├── metrics.py               # Hàm tính lỗi PER (Phoneme Error Rate)
│   ├── alignment.py             # Thuật toán gióng hàng Levenshtein Distance & G2P
│   ├── inference.py             # Bộ dự đoán âm vị trực tiếp từ âm thanh
│   └── utils/                   
│       ├── parse_textgrid.py    # Xử lý dọn bùn rác cho nhãn L2-ARCTIC .TextGrid
│       └── preprocess_audio.py  # Chuẩn hóa Audio
│
├── scripts/                     # 🚀 Các Kịch bản Chạy tự động (Execution Scripts)
│   ├── build_dataset.py         # Trích xuất metadata từ hàng nghìn file TextGrid
│   ├── split_dataset.py         # Chia tập Train/Val/Test
│   ├── build_vocab.py           # Sinh bộ từ vựng 45 âm vị ARPAbet sạch
│   ├── train.py                 # Kịch bản Huấn luyện Mô hình
│   └── debug_ctc.py             # Script chẩn đoán lỗi Collapse CTC Loss
│
├── app.py                       # 🎯 File ứng dụng chính chạy Inference toàn hệ thống
├── requirements.txt             # Danh sách thư viện Python cần thiết
└── README.md                    # Tài liệu hướng dẫn
```

---

## 🛠️ Hướng dẫn Cài đặt (Installation)

1. **Clone repository này về:**
```bash
git clone https://github.com/trungkien1511/pronunciation-assessment.git
cd pronunciation-assessment
```

2. **Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```
*(Lưu ý: Bạn nên cài đặt Pytorch bản hỗ trợ GPU CUDA để quá trình Huấn luyện (Training) diễn ra nhanh hơn).*

---

## 🚀 Hướng dấn Sử dụng (Usage)

### 1. Đánh giá file Âm thanh bất kỳ (Inference)
Sau khi cài đặt hoặc có Model đã train, bạn có thể tự thu âm một câu tiếng Anh `.wav` bất kỳ của bạn, và dùng App để chấm điểm:
```bash
python app.py --audio "đường_dẫn_đến_file_của_bạn.wav" --text "Câu tiếng anh mà bạn phát âm"
```
**Ví dụ:**
```bash
python app.py --audio "C:/Audio/Hello_World.wav" --text "Hello world"
```
*Kết quả sẽ trả về Danh sách các Lỗi ❌, 🔴, ⚠️ và Tổng số Điểm 100.*

---

### 2. Huấn luyện lại từ đầu (Training)
Nếu bạn tải bộ dữ liệu L2-ARCTIC về và muốn tự tay Train lại mô hình:
```bash
# Bước 1: Build Metadata
python scripts/build_dataset.py

# Bước 2: Sinh tập Train/Val/Test
python scripts/split_dataset.py

# Bước 3: Tạo Từ vựng (45 ARPAbet)
python scripts/build_vocab.py

# Bước 4: Chạy quá trình Fine-Tuning
python scripts/train.py
```

---

## ⚙ Tính năng Kỹ thuật nổi bật
* Khắc phục hoàn toàn lỗi **Catastrophic Forgetting & CTC Blank Collapse** (Loss kẹt ở 7.0) bằng cách dùng cơ chế **Transfer Learning** (đóng băng 90M đỉnh tri thức Transformer của `facebook/wav2vec2-base-960h`) và chỉ train chóp Classifier Head 45 classes (Vocab).
* Tự động nhận diện Âm tiết tiếng ồn `noise`, `silence`, xoá bỏ trọng âm rác để tinh chỉnh bảng từ vựng từ 93 xuống mức hoàn hảo 45 Phonemes.
* Cơ chế gióng hàng thông minh bỏ qua lỗi "Insertion" (Thừa âm) do người dùng chèn âm bậy vào giữa chuỗi.