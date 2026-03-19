import json
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

def initialize_model(vocab_path, model_name="facebook/wav2vec2-base"):
    """
    Khởi tạo mô hình Wav2Vec2 dành cho task CTC (Speech-to-Phoneme).
    Mod lại lớp Classification Head cuối cùng bằng Vocab Size của Phonemes hiện tại.
    
    Args:
        vocab_path (str): File chứa từ điển vocab.json.
        model_name (str): ID tải mô hình từ HuggingFace (mặc định Wav2Vec2-Base).
        
    Returns:
        Wav2Vec2ForCTC: Mô hình đã cấu hình.
    """
    # 1. Tải bộ Vocabulary để biết kích thước Vocab (đầu ra của Model)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    vocab_size = len(vocab)
    print(f"Vocab size hiện tại: {vocab_size}")
    
    # 2. Định nghĩa cấu hình padding token (CTC cần biết token nào là token đệm)
    pad_token_id = vocab.get("<pad>", 0)
    
    print(f"Khởi tạo mô hình '{model_name}' với classification head: {vocab_size} output classes...")
    
    # 3. Khởi tạo Mô hình Wav2Vec2 cho CTC
    # - Các trọng số Layer bên dưới (CNN layers extracts features) bắt đầu từ checkpoint gốc.
    # - Lớp Fully Connected trên cùng (Classification layer) sẽ được khởi tạo mới ngẫu nhiên 
    #   (phù hợp với vocab_size = 93 labels mới thay vì 30 chữ cái tiếng Anh cũ của hãng Meta).
    
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        attention_dropout=0.1,    # Ngăn overfitting
        hidden_dropout=0.1,       # Ngăn overfitting
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,      # Kỹ thuật SpecAugment: Bịt ngẫu nhiên thời gian để model học tốt hơn
        layerdrop=0.1,            # Giúp mô hình linh hoạt hơn
        ctc_loss_reduction="mean", # Loại Loss
        pad_token_id=pad_token_id, 
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True # Bắt buộc thêm dòng này để cho phép thay đầu não từ 32 chữ sang 45 chữ
    )
    
    # Freeze toàn bộ mô hình gốc (Cả CNN và Transformer)
    # Vì bản '960h' đã quá thông minh nghe tiếng Anh rồi, nếu thả cửa cho train tiếp 90 triệu tham số này 
    # với dataset nhỏ, Gradient khổng lồ sẽ phá nát trí nhớ của nó gây kẹt loss ở 7.0
    model.freeze_feature_encoder()
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
        
    print("Đã đóng băng (Freeze) toàn bộ Bộ Nguồn (CNN + Transformer). Chỉ train cái Đầu Mới (Classifier Head)!")
    
    return model

if __name__ == "__main__":
    vocab_json = r"d:\test\dataset_splits\vocab.json"
    
    try:
        # Chạy khởi tạo
        model = initialize_model(vocab_json, model_name="facebook/wav2vec2-base")
        
        # In ra tham số để confirm layer cuối cùng đổi từ 32 thành 93 (vocab_size)
        print("\n=== Model Initialized ===")
        print(f"Kiến trúc Classifier Layer cuối: {model.lm_head}")
        print(f"Tổng số tham số: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Triệu tham số")
    except Exception as e:
        print(f"Lỗi khởi tạo mô hình: {e}")
