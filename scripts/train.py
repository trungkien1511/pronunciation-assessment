import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainingArguments, Trainer
from phoneme_assessment.dataset import L2ArcticPhonemeDataset, DataCollatorCTCWithPadding
from phoneme_assessment.model import initialize_model
from phoneme_assessment.metrics import compute_metrics

def main():
    # 1. Đường dẫn dữ liệu
    data_dir = r"d:\test\dataset_splits"
    train_json = os.path.join(data_dir, "train.json")
    val_json = os.path.join(data_dir, "val.json")
    vocab_json = os.path.join(data_dir, "vocab.json")
    
    # 2. Khởi tạo Datasets
    print("Đang chuẩn bị dữ liệu...")
    train_dataset = L2ArcticPhonemeDataset(train_json, vocab_json)
    val_dataset = L2ArcticPhonemeDataset(val_json, vocab_json)
    
    # Ở đây do Wav2Vec2 tự xử lý Padding = 0 cho Input, 
    # và ta cần Padding Label = -100 để Pytorch CrossEntropy/CTC Loss bỏ qua.
    data_collator = DataCollatorCTCWithPadding(pad_token_id=0)
    
    # 3. Khởi tạo Models
    print("Đang khởi tạo mô hình Wav2Vec2-Base-960h...")
    model = initialize_model(vocab_json, "facebook/wav2vec2-base-960h")
    
    # 4. Thiết lập tham số huấn luyện (Hyperparameters)
    # Lượng RAM và VRAM sẽ tùy thuộc vào cấu hình máy, bạn có thể chỉnh batch_size lại nếu báo lỗi OOM (Out of Memory).
    training_args = TrainingArguments(
        output_dir=r"d:\test\wav2vec2-l2arctic",    # Thư mục lưu Checkpoints
        per_device_train_batch_size=8,              # Batch size lúc train. Nếu CPU/GPU yếu hãy hạ xuống 4 hoặc 2.
        per_device_eval_batch_size=8,               # Batch size lúc đánh giá.
        gradient_accumulation_steps=2,              # Gộp gradient (2 x 8 = 16 batch size thực tế)
        eval_strategy="steps",                      # Đánh giá sau mỗi N bước
        eval_steps=200,                             # N = 200
        save_strategy="steps",                      
        save_steps=400,                             # Lưu model sau mỗi 400 bước
        logging_steps=50,                           # In log ra terminal mỗi 50 bước
        learning_rate=1e-4,                         # Tốc độ học phù hợp cho dataset nhỏ (~2800 samples)
        warmup_steps=400,                           # Warmup steps giúp model ổn định lúc đầu
        save_total_limit=2,                         # Chỉ giữ tối đa 2 checkpoints gần nhất cho đỡ tốn bộ nhớ
        num_train_epochs=60,                        # Chạy 60 epochs để mô hình đủ khôn nhận diện được Phonemes
        fp16=torch.cuda.is_available(),             # Tự động dùng FP16(Mixed-Precision) nếu có GPU NVIDIA giúp train nhanh gấp đôi
        dataloader_num_workers=2,                   # Tăng tốc độ load data bằng multiprocessing
        load_best_model_at_end=True,                # Cuối cùng tự load cái model có metric tốt nhất
        metric_for_best_model="per",                # Chỉ số để tự động chọn model tốt nhất là Đoạn PER thấp nhất
        greater_is_better=False                     # PER càng thấp càng tốt
    )
    
    # 5. Khởi tạo Trainer HuggingFace
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 6. Kích hoạt Training
    print("\n🚀 Bắt đầu quá trình Huấn Luyện (Training)...")
    trainer.train()
    
    # 7. Lưu lại Model cuối cùng
    print("\n✅ Huấn luyện hoàn tất. Đang lưu mô hình...")
    final_model_path = r"d:\test\wav2vec2-l2arctic_final"
    trainer.save_model(final_model_path)
    
    print(f"Mô hình đã được lưu thành công tại: {final_model_path}")

if __name__ == "__main__":
    main()
