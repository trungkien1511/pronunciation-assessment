# 🎙️ L2-ARCTIC Phoneme-Level Mispronunciation Detection

Hệ thống AI chuyên biệt nhằm Đánh giá và Phát hiện Lỗi phát âm Tiếng Anh ở **cấp độ Âm vị (Phoneme-level)**. Dự án sử dụng mô hình học sâu **Wav2Vec2-Base-960h** kết hợp với thuật toán gióng hàng **Levenshtein Distance** và tự điển **G2P**, giúp phát hiện chính xác 3 loại lỗi phát âm thường gặp của người học tiếng Anh:
* ❌ **Substitution:** Đọc sai âm (VD: Phát âm `TH` thành `S` hoặc `D`, `NG` thành `N`).
* ⚠️ **Deletion:** Nuốt âm (VD: Quên đọc âm đuôi `S`, `T`, `L`).
* 🔴 **Insertion:** Đọc thừa âm (Chèn thêm âm rác vào từ).

Hệ thống được thiết kế theo kiến trúc chuẩn MLOps, tinh chỉnh (Fine-tuning) khéo léo từ bộ tệp dữ liệu âm thanh người học tiếng Anh [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/).

---

## 📂 Kiến trúc Mã nguồn (Repository Structure)

Dự án được phân chia module rõ ràng để tái sử dụng và dễ dàng bảo trì:

```text
├── src/          # 🧠 Package Lõi (Core Backend)
│   ├── dataset.py               # Xử lý Pytorch Dataset & Padding (DataCollator CTC)
│   ├── model.py                 # Khởi tạo mô hình & Logic Rã đông (Gradual Unfreezing)
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
│   ├── train.py                 # (Finetune Vỏ) Kịch bản Huấn luyện Mô hình phân loại
│   ├── finetune.py              # (Finetune Sâu) Kịch bản Rã đông Transformer siêu vi
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
# Test bằng bản Final siêu việt (Nếu bạn đã chạy lệnh rã đông):
python app.py --audio "đường_dẫn.wav" --text "Câu nói" --model_dir "wav2vec2-l2arctic_finetuned"

# Hoặc test bản mặc định:
python app.py --audio "C:/Audio/mangos.wav" --text "I eat a mango" --model_dir "wav2vec2-l2arctic_final"
```

*Kết quả sẽ trả về Danh sách các Lỗi ❌, 🔴, ⚠️ và Tổng số Điểm 100/100 có độ chính xác State-of-the-Art.*

---

### 2. Huấn luyện lại từ đầu (Training Pipeline)
Bộ kịch bản đã được chúng tôi thiết kế liên hoàn từ A -> Z. Nếu bạn muốn tự tay Build lại bộ Não AI:

```bash
# Bước 1: Build Metadata
python scripts/build_dataset.py

# Bước 2: Sinh tập Train/Val/Test
python scripts/split_dataset.py

# Bước 3: Tạo Từ vựng (45 ARPAbet) cực sạch
python scripts/build_vocab.py

# Bước 4: Chạy quá trình Đúc Lớp Vỏ (Classifier Head)
# Quá trình này sẽ đóng băng 90M đỉnh Transformer để mô hình học quen mặt chữ (Train ~60 Epochs)
python scripts/train.py

# Bước 5: (TỐI ƯU CẤP CAO) Rã Đông Từng Phần (Gradual Unfreezing)
# Mở khóa 2 lớp não sâu, dùng tốc độ siêu vi 1e-5 để AI hiểu giọng vùng miền L2
python scripts/finetune.py
```
