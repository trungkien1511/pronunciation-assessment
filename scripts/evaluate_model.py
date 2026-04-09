import os
import torch
import evaluate
from transformers import Trainer, TrainingArguments
import sys

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dataset import L2ArcticPhonemeDataset, DataCollatorCTCWithPadding
from core.model import load_finetuned_model
from core.metrics import compute_metrics
from transformers import Wav2Vec2ForCTC

def evaluate_model():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "wav2vec2-l2arctic_finetuned_v3"
    test_json = r"dataset_splits\test.json"
    vocab_json = r"dataset_splits\vocab.json"
    
    if not os.path.exists(model_dir):
        print(f"Lỗi: Không tìm thấy thư mục model {model_dir}")
        return
        
    print(f"Loading model from: {model_dir}...")
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    
    print("Loading Test data...")
    # augment=False for evaluation
    test_dataset = L2ArcticPhonemeDataset(test_json, vocab_json, augment=False)
    data_collator = DataCollatorCTCWithPadding(pad_token_id=0)
    
    # Simple training args just for eval
    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=8,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, vocab_json),
    )
    
    print("\nStarting model evaluation on Test set...")
    results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Model Directory : {model_dir}")
    print(f"Test Samples    : {len(test_dataset)}")
    metric_per = results.get("eval_per", 1.0)
    print(f"Phoneme Error Rate (PER): {metric_per*100:.2f}%")
    print(f"Accuracy: {(1.0 - metric_per)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
