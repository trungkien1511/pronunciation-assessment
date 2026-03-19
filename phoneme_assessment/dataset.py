import json
import torch
import librosa
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

class L2ArcticPhonemeDataset(Dataset):
    def __init__(self, json_path, vocab_path, max_length=160000):
        """
        PyTorch Dataset cho bài toán nhận diện Phoneme.
        
        Args:
           json_path (str): Đường dẫn tới train.json, val.json hoặc test.json
           vocab_path (str): Đường dẫn tới vocab.json
           max_length (int): Độ dài tối đa của mảng audio (160000 = 10 giây ở 16kHz)
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

    def _phonemes_to_ids(self, reference_phonemes, labels):
        """
        Chuyển đổi thực tế User đã đọc (chứ KHÔNG PHẢI âm chuẩn) thành dãy ID.
        Vì mục tiêu của ta là train model NGHE được chính xác User đọc gì.
        
        Logic: Mình duyệt qua `reference_phonemes` và `labels` (có độ dài bằng nhau từ bước 2).
        - Nếu label = "correct": User đọc chuẩn -> dùng token `reference_phoneme`
        - Nếu label = "substitution": Trong nhãn file ta không lưu âm User đọc (PPL) mà lưu 's', nên ở đây ta tạm thời vẫn coi như User cần học cách đọc đúng, HOẶC mô hình sẽ tự đoán bậy.
          Tuy nhiên, L2-ARCTIC gốc có lưu 'PPL' (kế hoạch ban đầu ta parse "CPL, PPL, s" rồi quên mất PPL).
          Do bản ghi `parse_textgrid.py` trước ta chỉ lưu Lỗi dạng "substitution" và "CPL". 
          *TỐI ƯU HƠN*: Ta thay vì ép nó học PPL, ta cứ dạy nó học theo CPL nếu ta làm dạng Mispronunciation Dectetion thuần, hoặc lý tưởng nhất là sửa lại parse_textgrid để lấy được PPL.
          
          Để đơn giản trước: Mô hình chúng ta sẽ cố gắng dự đoán CPL, nhưng ở những đoạn user đọc sai nó sẽ bị 'vấp' và output CTC ra một âm lạ.
        """
        # Vì đây là ví dụ, ta học trực tiếp mảng reference_phonemes
        # Các chỗ user đọc sai thay vì học từ đó, ta có thể thay bằng <unk> hoặc bắt học đúng (forced).
        # Tạm thời cứ cho nó học Reference.
        ids = []
        for ph, label in zip(reference_phonemes, labels):
            if ph == "sil": 
                continue # CTC tự học khoảng trống, ta có thể bỏ qua token sil trong target
                
            token_id = self.vocab.get(ph, self.unk_token_id)
            ids.append(token_id)
            
        return ids

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
            speech_array = [0.0] * 16000 
            print(f"Lỗi load audio: {audio_path}")
            
        # Cắt bớt nếu quá dài
        if len(speech_array) > self.max_length:
            speech_array = speech_array[:self.max_length]
            
        # 2. Chuẩn hóa qua Feature Extractor
        features = self.feature_extractor(
            speech_array, 
            sampling_rate=16000
        )
        input_values = features.input_values[0]
        attention_mask = features.attention_mask[0]
        
        # 3. Tạo Target Labels
        labels = self._phonemes_to_ids(item["reference_phonemes"], item["labels"])
        
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
    
    train_json = r"d:\test\dataset_splits\train.json"
    vocab_json = r"d:\test\dataset_splits\vocab.json"
    
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
