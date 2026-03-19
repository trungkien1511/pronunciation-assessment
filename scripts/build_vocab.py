import json
import os

def build_vocab(train_json, output_vocab):
    """
    Xây dựng bộ từ vựng (Vocabulary) từ tập Train.
    
    Args:
        train_json (str): Đường dẫn đến file train.json
        output_vocab (str): Đường dẫn lưu file vocab.json
    """
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    # Dùng set để lấy các phoneme duy nhất
    unique_phonemes = set()
    
    for item in train_data:
        for ph in item['reference_phonemes']:
            unique_phonemes.add(ph)
            
    # Sắp xếp alphabet cho dễ nhìn
    unique_phonemes = sorted(list(unique_phonemes))
    
    # Tạo dictionary vocab
    # Thêm các special tokens bắt buộc cho Wav2Vec2 / CTC
    vocab_dict = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4  # Dùng làm word boundary nếu cần
    }
    
    # Đánh số tiếp các phoneme
    start_idx = len(vocab_dict)
    for i, ph in enumerate(unique_phonemes):
        vocab_dict[ph] = start_idx + i
        
    # Lưu ra file
    os.makedirs(os.path.dirname(output_vocab) if os.path.dirname(output_vocab) else '.', exist_ok=True)
    with open(output_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=4, ensure_ascii=False)
        
    print(f"Đã tạo Vocab gồm {len(vocab_dict)} tokens (bao gồm special tokens).")
    print(f"Các tokens: {list(vocab_dict.keys())}")
    print(f"File lưu tại: {output_vocab}")

if __name__ == "__main__":
    train_file = r"d:\test\dataset_splits\train.json"
    vocab_file = r"d:\test\dataset_splits\vocab.json"
    
    if os.path.exists(train_file):
        build_vocab(train_file, vocab_file)
    else:
        print(f"Không tìm thấy file {train_file}. Hãy chạy split_dataset.py trước.")
