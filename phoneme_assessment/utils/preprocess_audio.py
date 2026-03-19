import os
import librosa
import soundfile as sf
from pathlib import Path

def preprocess_audio(input_path, output_path=None, target_sr=16000):
    """
    Tiền xử lý file Audio: 
    1. Resampling về target_sr (mặc định 16kHz).
    2. Chuyển thành Mono (1 channel).
    3. (Tùy chọn) Lưu file mới nếu có output_path.
    
    Args:
        input_path (str): Đường dẫn đến file .wav gốc (thường là 44.1kHz stereo/mono).
        output_path (str, optional): Nơi lưu file sau khi xử lý. Nếu None, chỉ trả về biến audio.
        target_sr (int): Tần số lấy mẫu mục tiêu.
        
    Returns:
        tuple(np.ndarray, int): Mảng tín hiệu âm thanh và tần số lấy mẫu (sr).
    """
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file {input_path}")
        return None, None
        
    try:
        # librosa.load tự động convert sang Mono (mono=True mặc định)
        # và tự động Resample về sr=target_sr
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        # Lưu file nếu cần thiết (để train sau này không cần load lại thư viện nặng)
        if output_path is not None:
            # Tạo thư mục con nếu chưa có
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sr)
            
        return audio, sr
        
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý audio {input_path}: {e}")
        return None, None

def batch_preprocess_directory(input_dir, output_dir, target_sr=16000):
    """
    Tiết xử lý hàng loạt mọi file wav trong 1 thư mục.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    wav_files = list(input_path.glob('**/*.wav'))
    print(f"Tìm thấy {len(wav_files)} file .wav trong {input_dir}")
    
    for i, wav_file in enumerate(wav_files):
        # Giữ nguyên cấu trúc thư mục
        rel_path = wav_file.relative_to(input_path)
        out_file = output_path / rel_path
        
        preprocess_audio(str(wav_file), str(out_file), target_sr)
        
        if (i + 1) % 100 == 0:
            print(f"Đã xử lý {i + 1}/{len(wav_files)} files...")
    
    print("Hoàn tất tiền xử lý audio!")

if __name__ == "__main__":
    # Test thử 1 file
    test_wav = "sample.wav"
    if os.path.exists(test_wav):
        print(f"Đang xử lý {test_wav}")
        audio_data, sr = preprocess_audio(test_wav, "processed_sample.wav")
        print(f"Audio shape: {audio_data.shape}, Sample rate: {sr}")
    else:
        print(f"File test {test_wav} không tồn tại.")
