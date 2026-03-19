import os
import json
import librosa
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom functions
from phoneme_assessment.utils.parse_textgrid import parse_textgrid
from phoneme_assessment.utils.preprocess_audio import preprocess_audio

def build_dataset_metadata(l2_arctic_dir, output_json, sample_rate=16000):
    """
    Duyệt toàn bộ thư mục L2-ARCTIC để khớp nối Audio và TextGrid.
    Tạo metadata (JSON) tổng hợp.
    
    Args:
        l2_arctic_dir (str): Folder gốc chứa dữ liệu (chứa các thư mục như ABA, SKA, ...).
        output_json (str): File JSON đầu ra.
        sample_rate (int): SR mục tiêu (dùng để tính thời lượng file wav).
    """
    root_path = Path(l2_arctic_dir)
    metadata_list = []
    error_count = 0
    
    # Tìm tất cả thư mục cấp 1 (Tên các speaker, vd: ABA, ASI, BWC...)
    speakers = [d for d in root_path.iterdir() if d.is_dir() and d.name != 'suitcase_corpus']
    
    print(f"Tìm thấy {len(speakers)} thư mục speakers.")
    
    for speaker_dir in tqdm(speakers, desc="Processing speakers"):
        speaker_id = speaker_dir.name
        
        # Đường dẫn tới thư mục wav và annotation
        wav_dir = speaker_dir / 'wav'
        annotation_dir = speaker_dir / 'annotation'
        
        if not wav_dir.exists() or not annotation_dir.exists():
            continue
            
        # Tìm tất cả các file TextGrid đã được annotate
        for tg_path in annotation_dir.glob('*.TextGrid'):
            file_id = tg_path.stem  # vd: arctic_a0001
            wav_path = wav_dir / f"{file_id}.wav"
            
            # Bỏ qua nếu không có file wav tương ứng
            if not wav_path.exists():
                error_count += 1
                continue
                
            # 1. Trích xuất Chuẩn & Lỗi từ TextGrid
            reference_phonemes, labels = parse_textgrid(str(tg_path))
            if not reference_phonemes:
                error_count += 1
                continue
                
            # 2. Lấy thời lượng từ Audio (Chỉ đọc header nhanh bằng librosa.get_duration)
            try:
                duration = librosa.get_duration(path=str(wav_path))
            except Exception as e:
                print(f"Lỗi khi đọc duration file {wav_path}: {e}")
                error_count += 1
                continue
                
            # 3. Gom ráp vào Dictionary
            sample = {
                "id": f"{speaker_id}_{file_id}",
                "speaker_id": speaker_id,
                "audio_filepath": str(wav_path.absolute()),
                "textgrid_filepath": str(tg_path.absolute()),
                "reference_phonemes": reference_phonemes,
                "labels": labels,
                "duration": round(duration, 3)
            }
            metadata_list.append(sample)
            
    # Lưu vào file JSON
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Hoàn tất! Đã tạo thành công {len(metadata_list)} bản ghi metadata.")
    print(f"Đã bỏ qua {error_count} file bị lỗi ghép cặp hoặc hỏng.")
    print(f"File được lưu tại: {output_json}")

if __name__ == "__main__":
    # Thay đổi đường dẫn này trỏ tới bộ dữ liệu thực tế l2arctic_release_v5.0 trên máy bạn
    dataset_dir = r"d:\test\l2arctic_release_v5.0"
    output_file = r"d:\test\train_metadata.json"
    
    if os.path.exists(dataset_dir):
        print(f"Bắt đầu xử lý thư mục {dataset_dir} ...")
        build_dataset_metadata(dataset_dir, output_file)
    else:
        print(f"Không tìm thấy thư mục dữ liệu l2arctic ở {dataset_dir}")
