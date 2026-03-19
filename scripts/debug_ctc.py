import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from phoneme_assessment.dataset import L2ArcticPhonemeDataset, DataCollatorCTCWithPadding
from phoneme_assessment.model import initialize_model
import json

def debug():
    vocab_json = r"d:\test\dataset_splits\vocab.json"
    train_json = r"d:\test\dataset_splits\train.json"
    
    print("1. Loading dataset...")
    dataset = L2ArcticPhonemeDataset(train_json, vocab_json)
    collator = DataCollatorCTCWithPadding(pad_token_id=0)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(dataloader))
    
    print("\n2. Initializing Model...")
    # Thử kích hoạt ctc_zero_infinity
    model = initialize_model(vocab_json, "facebook/wav2vec2-base-960h")
    model.config.ctc_zero_infinity = True
    model.train()
    
    input_values = batch["input_values"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)
    
    print(f"\n3. Batch Inputs Shapes:")
    print(f"input_values: {input_values.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"labels: {labels.shape}")
    
    print("\n4. Labels content (first sample):")
    label_0 = labels[0]
    valid_labels = label_0[label_0 != -100]
    print(f"Valid length: {len(valid_labels)}")
    print(f"Tokens: {valid_labels.tolist()}")
    
    print("\n5. Forward pass...")
    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels
    )
    
    logits = outputs.logits
    loss = outputs.loss
    
    print(f"Logits shape: {logits.shape}")
    print(f"Calculated Target Lengths (internal by HF): {len(valid_labels)}")
    print(f"Calculated Input Lengths (internal by HF): {model._get_feat_extract_output_lengths(attention_mask.sum(-1))[0].item()}")
    print(f"Initial Loss for batch: {loss.item()}")
    
    print("\n6. Checking Logits Distribution...")
    probs = torch.nn.functional.softmax(logits[0, 0, :], dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)
    print("Probabilities of first frame:")
    for p, i in zip(top_probs, top_indices):
        print(f"  Token {i.item()}: {p.item():.4f}")
        
    print("\n7. Backpropagate...")
    loss.backward()
    
    # Check max grad in lm_head
    grad_norm = model.lm_head.weight.grad.norm().item()
    print(f"LM Head Grad Norm: {grad_norm}")
    print("Done!")

if __name__ == "__main__":
    debug()
