import torch
import librosa
import numpy as np
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
import json

class L2ArcticInference:
    def __init__(self, model_path, vocab_path="d:/test/dataset_splits/vocab.json"):
        """
        Khởi tạo pipeline Inference (Dự đoán) của mô hình.
        
        Args:
            model_path (str): Đường dẫn tới thư mục model đã train (vd: 'd:/test/wav2vec2-l2arctic_final')
            vocab_path (str): File từ điển vocab.json để biết map ID ra chữ cái
        """
        print(f"Đang tải mô hình từ: {model_path} ...")
        # Load Model và Feature Extractor chuẩn
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        
        # Tự động đẩy model lên GPU nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Bật chế độ inference (tắt dropout)
        print(f"Mô hình đang chạy trên thiết bị: {self.device}")
        
        # Load Vocab và đảo ngược từ điển (ID -> Phoneme)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.id_to_ph = {v: k for k, v in vocab.items()}

    def predict(self, audio_path):
        """
        Đưa 1 file âm thanh thực tế qua mô hình để lấy ra chuỗi Phoneme nhận diện được.
        """
        # 1. Tiền xử lý âm thanh về chuẩn 16000Hz, Mono
        speech_array, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # 2. Extract Features (Bơm tensor về format mà mô hình cần)
        inputs = self.feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        # 3. Chạy qua mô hình (Sử dụng torch.no_grad() để tắt tính toán Gradient, tăng tốc độ)
        with torch.no_grad():
            logits = self.model(input_values).logits
            
        # 4. Giải mã ma trận kết quả
        # Lấy giá trị lớn nhất theo chiều -1 (Argmax) để ra kết quả ID
        predicted_ids = torch.argmax(logits, dim=-1)[0]
        
        # 5. Dịch ID về dạng Text (giống kỹ thuật trong metrics.py)
        predicted_phonemes = []
        prev_ph = None
        
        for idx in predicted_ids:
            ph = self.id_to_ph.get(int(idx), "")
            
            # Luật CTC: Bỏ qua <pad> và các token lặp nhau liền kề
            if ph in ["<pad>", "<s>", "</s>"]:
                prev_ph = ph
                continue
            if ph == prev_ph:
                continue
            if ph and ph != "|":
                predicted_phonemes.append(ph)
            prev_ph = ph
            
        return predicted_phonemes

if __name__ == "__main__":
    print("Script sẵn sàng cho module Inference!")
    model_dir = r"d:\test\wav2vec2-l2arctic_final" # Trỏ thẳng đến THƯ MỤC này
    
    if os.path.exists(model_dir):
        ai = L2ArcticInference(model_dir)
        
        # Test thử AI nghe một file wav có sẵn trong tập Test
        # Đổi đường dẫn này tới một file wav tồn tại trong máy bạn
        test_audio = r"d:\test\l2arctic_release_v5.0\ABA\wav\arctic_a0001.wav" 
        
        if os.path.exists(test_audio):
            predicted = ai.predict(test_audio)
            print("AI nghe được:", predicted)
        else:
            print(f"Không tìm thấy file Test Audio: {test_audio}")
    else:
        print(f"Chưa có model ở thư mục {model_dir}")
