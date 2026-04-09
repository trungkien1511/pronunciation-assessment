import argparse
import os
import sys
import json

# Trỏ đường dẫn path về thư mục cha để có thể import từ gốc của module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service import pronunciation_service

def main():
    parser = argparse.ArgumentParser(description="Test Pronunciation Module AI")
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn đến file Audio (.wav)")
    parser.add_argument("--text", type=str, required=True, help="Văn bản tiếng Anh cần đọc")
    args = parser.parse_args()

    # Nạp model
    if not pronunciation_service.load_model():
        print("Không thể bật module.")
        return

    if not os.path.exists(args.audio):
        print(f"Lỗi: Không tìm thấy file âm thanh - {args.audio}")
        return

    # Process and get JSON response
    try:
        print(f"\n🎧 Đang phân tích file: {args.audio}")
        print(f"📝 Dựa trên câu: '{args.text}'\n")
        
        result_json = pronunciation_service.process_audio(args.audio, args.text)
        
        # In JSON đẹp mắt ra màn hình
        print(json.dumps(result_json, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print("LỖI XỬ LÝ:", str(e))

if __name__ == "__main__":
    main()
