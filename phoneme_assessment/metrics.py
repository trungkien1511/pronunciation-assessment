import json
import numpy as np
import evaluate

def compute_metrics(pred, vocab_path="d:/test/dataset_splits/vocab.json"):
    """
    Hàm tính toán Phoneme Error Rate (PER) dựa trên model predictions.
    Sẽ được HuggingFace Trainer tự động gọi sau mỗi Epoch (hoặc eval step).
    
    Args:
        pred: Một đối tượng EvaluatePrediction chứa `predictions` (logits) và `label_ids`.
        vocab_path: Đường dẫn tới file vocab để mapping ID về chữ cái.
        
    Returns:
        dict: Chứa giá trị PER {"per": 0.xxxx}
    """
    # Load Vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    # Tạo từ điển ngược ID -> Phoneme (ví dụ: 15 -> 'F')
    id_to_ph = {v: k for k, v in vocab.items()}
    
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    label_ids = pred.label_ids
    
    # Ở Padding Labels (lúc mình code collator), mình dùng giá trị -100 để Pytorch bơ đi.
    # Khi Decode ra list string để tính PER, mình cần đổi -100 về giá trị pad_token_id (0) tạm thời.
    label_ids[label_ids == -100] = vocab.get("<pad>", 0)

    # Giải mã (Decode) ID về Chuỗi Phoneme Text
    # Hàm này mô phỏng Wav2Vec2CTCTokenizer.decode (nhưng đơn giản hơn 1 chút)
    
    def decode_ids(ids_list, is_pred=False):
        decoded_strings = []
        for sequence in ids_list:
            ph_sequence = []
            prev_ph = None
            for idx in sequence:
                idx = int(idx)
                ph = id_to_ph.get(idx, "")
                
                # Logic của CTC: 
                # - Bỏ qua Pad Token và các Token lặp lại giống hệt nhau liên tiếp.
                # - <pad> token sẽ tự phân tách các ký tự lặp hợp lệ.
                if is_pred:
                    if ph == "<pad>" or ph == "<s>" or ph == "</s>":
                        prev_ph = ph
                        continue
                    if ph == prev_ph:
                        continue
                        
                else: 
                     # Nếu là label gốc thì chỉ bỏ qua special tokens
                     if ph == "<pad>" or ph == "<s>" or ph == "</s>":
                         continue
                
                if ph and ph != "|": # Ký tự ranh giới từ '|' cũng là special format
                     ph_sequence.append(ph)
                     
                prev_ph = ph
                
            # Jiwer yêu cầu input là String dạng text (các từ/phonemes cách nhau bằng dấu cách)
            decoded_strings.append(" ".join(ph_sequence))
            
        return decoded_strings

    pred_str = decode_ids(pred_ids, is_pred=True)
    label_str = decode_ids(label_ids, is_pred=False)
    
    # Sử dụng evaluate library với metrics "wer" (Word Error Rate),
    # Do input ta truyền vào là các Phoneme cách nhau bởi dấu cách (ví dụ: "T H I NG K"), 
    # nên WER ở đây về bản chất TOÁN HỌC chính là PER (Phoneme Error Rate).
    metric = evaluate.load("wer")
    
    per = metric.compute(predictions=pred_str, references=label_str)

    return {"per": per}

if __name__ == "__main__":
    # Test Hàm Tính Metric nhanh
    import evaluate
    try:
         evaluate.load("wer")
         print("Đã tải module WER Metric thành công từ Evaluate.")
    except Exception as e:
         print(f"Lỗi: {e}. Vui lòng chạy pip install evaluate jiwer")
