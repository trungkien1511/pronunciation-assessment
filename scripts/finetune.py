import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainingArguments, Trainer
from src.dataset import L2ArcticPhonemeDataset, DataCollatorCTCWithPadding
from src.model import load_finetuned_model
from src.metrics import compute_metrics

def main():
    # 1. Đường dẫn dữ liệu và Model cũ đã train xong vỏ (Classifier)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_json = os.path.join(base_dir, "dataset_splits", "train.json")
    val_json = os.path.join(base_dir, "dataset_splits", "val.json")
    vocab_json = os.path.join(base_dir, "dataset_splits", "vocab.json")
    
    old_model_dir = os.path.join(base_dir, "wav2vec2-l2arctic_final_v3")
    output_dir = os.path.join(base_dir, "wav2vec2-l2arctic_finetuned_v3") # Thư mục lưu thành quả mới
    
    print("=========================================================================")
    print("🚀 GIAI ĐOẠN 5: TỐI ƯU CẤP ĐỘ CAO (GRADUAL UNFREEZING)")
    print("=========================================================================")
    
    # 2. Chuẩn bị DataLoader
    print("Đang chuẩn bị dữ liệu...")
    train_dataset = L2ArcticPhonemeDataset(train_json, vocab_json, augment=True)
    val_dataset = L2ArcticPhonemeDataset(val_json, vocab_json, augment=False)
    data_collator = DataCollatorCTCWithPadding(pad_token_id=0)
    
    # 3. Khởi tạo Model Rã Đông
    # Mở khóa 2 lớp Transformer cuối cùng để học cách bắt chất giọng người châu Á
    model = load_finetuned_model(old_model_dir, unfreeze_top_n_layers=4)
    
    # 4. Thiết lập Hyperparameters "Vi phẫu" (Micro-learning rate)
    # Vì đang đụng vào Não giữa (Transformer) vốn đã quá giỏi, phải xài Learning Rate siêu nhỏ
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-6,          # LR CỰC NHỎ: 5e-6 (do mở 4 layers, cần LR thấp hơn để tránh Catastrophic Forgetting)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,         # Chỉ rèn nhẹ 10 Epochs
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        dataloader_num_workers=0
    )
    
    # 5. Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, vocab_json),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=train_dataset.feature_extractor,
    )
    
    print("\n🔥 Bắt đầu quá trình Rã Đông & Huấn Luyện Sâu (Fine-Tuning)...")
    trainer.train()
    
    print("\n✅ Huấn luyện hoàn tất. Đang lưu mô hình siêu cấp...")
    trainer.save_model(output_dir)
    train_dataset.feature_extractor.save_pretrained(output_dir)
    print(f"🎉 Mô hình đã được rã đông và nâng cấp thành công tại: {output_dir}")

if __name__ == "__main__":
    main()
