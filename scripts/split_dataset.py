import json
import random
import os
from collections import defaultdict

def split_dataset(input_json, output_dir, test_speakers=None, val_ratio=0.1, random_seed=42):
    """
    Chia tập dữ liệu thành Train, Val, Test. Đảm bảo speaker-independent cho Test set.
    """
    random.seed(random_seed)
    
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Tổng số bản ghi ban đầu: {len(data)}")
    
    # 1. Lọc bỏ các bản ghi không hợp lệ hoặc nằm trong danh sách loại trừ
    excluded_sentences = ['arctic_b0013', 'arctic_a0158', 'arctic_b0398', 
                          'arctic_b0345', 'arctic_a0298', 'arctic_a0094']
                          
    valid_data = []
    for item in data:
        # Bỏ qua nếu là khoảng nghỉ dài hoặc file không đọc
        if any(ex in item['id'] for ex in excluded_sentences):
             continue
        if len(item['reference_phonemes']) < 3:
             continue
        valid_data.append(item)
        
    print(f"Sau khi lọc: {len(valid_data)} bản ghi hợp lệ.")

    # 2. Gom nhóm theo Speaker
    speaker_data = defaultdict(list)
    for item in valid_data:
        speaker_data[item['speaker_id']].append(item)
        
    speakers_list = list(speaker_data.keys())
    print(f"Tổng số speakers: {len(speakers_list)} -> {speakers_list}")
    
    # 3. Chọn Test Speakers (để đánh giá speaker-independent)
    if test_speakers is None:
        # Tự động chọn ngẫu nhiên 3 người (ví dụ: 1 nam, 1 nữ, đa dạng ngôn ngữ nếu có thể)
        test_speakers = random.sample(speakers_list, 3)
        
    print(f"Speakers dành riêng cho tập Test: {test_speakers}")
    
    # 4. Phân chia dữ liệu
    train_val_data = []
    test_data = []
    
    for spk, items in speaker_data.items():
        if spk in test_speakers:
            test_data.extend(items)
        else:
            train_val_data.extend(items)
            
    # Xáo trộn tập train_val
    random.shuffle(train_val_data)
    
    # Tách Validation ra từ Train
    val_size = int(len(train_val_data) * val_ratio)
    val_data = train_val_data[:val_size]
    train_data = train_val_data[val_size:]
    
    print(f"\nKết quả chia tập:")
    print(f"- Train set : {len(train_data)} samples")
    print(f"- Val set   : {len(val_data)} samples")
    print(f"- Test set  : {len(test_data)} samples")
    
    # 5. Lưu kết quả
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu {out_path}")

if __name__ == "__main__":
    input_file = r"d:\test\train_metadata.json"
    output_directory = r"d:\test\dataset_splits"
    
    if os.path.exists(input_file):
        # Có thể chủ động set test_speakers=['ABA', 'LXC', 'PNV'] nếu muốn cố định
        split_dataset(input_file, output_directory)
    else:
        print(f"Không tìm thấy file {input_file}")
