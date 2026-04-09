import json
import torch
import librosa
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

class L2ArcticPhonemeDataset(Dataset):
    def __init__(self, json_path, vocab_path, max_length=160000, augment=False):
        """
        PyTorch Dataset cho bài toán nhận diện Phoneme.
        
        Args:
           json_path (str): Đường dẫn tới train.json, val.json hoặc test.json
           vocab_path (str): Đường dẫn tới vocab.json
           max_length (int): Độ dài tối đa của mảng audio (160000 = 10 giây ở 16kHz)
           augment (bool): Bật/tắt Data Augmentation (chỉ bật khi training)
        """
        # Đọc file metadata
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Đọc từ điển vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        # Khởi tạo Feature Extractor giống như Model (chuẩn hóa zero mean / unit variance)
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, 
            sampling_rate=16000, 
            padding_value=0.0, 
            do_normalize=True, 
            return_attention_mask=True
        )
        
        self.max_length = max_length
        self.unk_token_id = self.vocab.get("<unk>", 3)
        self.augment = augment

    def _phonemes_to_ids(self, item):
        """
        Chuyển đổi danh sách phoneme thành dãy ID cho CTC target.
        
        Logic PPL (Perceived Phoneme Label):
        - Nếu label = "correct": Dùng reference_phoneme (CPL = PPL, đọc đúng)
        - Nếu label = "substitution": Dùng perceived_phoneme (PPL) - âm thực tế người nói đọc
          → Dạy model nghe đúng cái người nói phát ra, không ép nghe thành âm chuẩn
        - Nếu label = "deletion": Bỏ qua (CTC sẽ tự học khoảng trống)
        """
        reference_phonemes = item["reference_phonemes"]
        labels = item["labels"]
        # Backward compatible: dùng perceived_phonemes nếu có, nếu không fallback về reference
        perceived_phonemes = item.get("perceived_phonemes", None)
        
        ids = []
        for i, (ph, label) in enumerate(zip(reference_phonemes, labels)):
            if ph == "sil": 
                continue  # CTC tự học khoảng trống
            
            if label == "deletion":
                continue  # Người nói nuốt âm, không có output tương ứng
                
            if label == "substitution" and perceived_phonemes is not None:
                # Dùng PPL (âm thực tế người nói đọc) làm target
                ppl = perceived_phonemes[i]
                if ppl is not None:
                    token_id = self.vocab.get(ppl, self.unk_token_id)
                else:
                    token_id = self.vocab.get(ph, self.unk_token_id)
            else:
                # Correct hoặc fallback: dùng reference phoneme
                token_id = self.vocab.get(ph, self.unk_token_id)
                
            ids.append(token_id)
            
        return ids

    def _augment_audio(self, speech_array, sr=16000):
        """
        Áp dụng Data Augmentation cho audio. Mỗi kỹ thuật có 50% xác suất được áp dụng.
        
        1. Speed perturbation (0.9x ~ 1.1x)
        2. Volume perturbation (±20%)
        3. Additive Gaussian noise (SNR 20~40 dB)
        """
        # 1. Speed Perturbation
        if random.random() < 0.5:
            speed_factor = random.uniform(0.9, 1.1)
            speech_array = librosa.effects.time_stretch(speech_array, rate=speed_factor)
        
        # 2. Volume Perturbation
        if random.random() < 0.5:
            volume_factor = random.uniform(0.8, 1.2)
            speech_array = speech_array * volume_factor
            
        # 3. Additive Gaussian Noise (SNR 20~40 dB)
        if random.random() < 0.5:
            snr_db = random.uniform(20, 40)
            signal_power = np.mean(speech_array ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(speech_array))
            speech_array = speech_array + noise.astype(speech_array.dtype)
        
        # 4. Pitch Shifting (±2 semitones) - Giúp model quen với giọng nam trầm và nữ cao
        if random.random() < 0.3:
            n_steps = random.uniform(-2, 2)
            speech_array = librosa.effects.pitch_shift(speech_array, sr=sr, n_steps=n_steps)
            
        return speech_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Load Audio
        audio_path = item["audio_filepath"]
        try:
            # librosa sẽ trả về mono wav, sr16k (vì preprocess_audio trước đó đã convert sẵn)
            speech_array, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            # Fake data if error file
            speech_array = np.zeros(16000, dtype=np.float32)
            print(f"Lỗi load audio: {audio_path}")
            
        # 2. Data Augmentation (chỉ khi training)
        if self.augment:
            speech_array = self._augment_audio(speech_array)
            
        # Cắt bớt nếu quá dài
        if len(speech_array) > self.max_length:
            speech_array = speech_array[:self.max_length]
            
        # 3. Chuẩn hóa qua Feature Extractor
        features = self.feature_extractor(
            speech_array, 
            sampling_rate=16000
        )
        input_values = features.input_values[0]
        attention_mask = features.attention_mask[0]
        
        # 4. Tạo Target Labels (sử dụng PPL cho substitution cases)
        labels = self._phonemes_to_ids(item)
        
        return {
            "input_values": input_values,     # Float Tensor
            "attention_mask": attention_mask, # Int Tensor
            "labels": labels                  # Int Array
        }

# Data Collator (Hàm dùng để gộp Batch, do audio và label dài ngắn khác nhau)
class DataCollatorCTCWithPadding:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        
        # Padding input_values
        input_values = [torch.tensor(feature["input_values"]) for feature in features]
        # Pad sequence cho Input
        input_values_padded = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )
        
        # Padding attention_mask
        attention_mask = [torch.tensor(feature["attention_mask"]) for feature in features]
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        # Padding labels
        labels = [torch.tensor(feature["labels"]) for feature in features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100 # -100 để PyTorch CrossEntropy/CTC ignore
        )
        
        return {
            "input_values": input_values_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }

if __name__ == "__main__":
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    train_json = "dataset_splits\train.json"
    vocab_json = "dataset_splits\vocab.json"
    
    dataset = L2ArcticPhonemeDataset(train_json, vocab_json)
    collator = DataCollatorCTCWithPadding(pad_token_id=0)
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    
    print(f"Tổng số mẫu trong Dataset: {len(dataset)}")
    
    # Lấy thử 1 batch
    for batch in dataloader:
        print("Input Values Shape:", batch["input_values"].shape)
        print("Labels Shape:", batch["labels"].shape)
        print("Sample Labels:", batch["labels"][0])
        break
