import os
from phoneme_assessment.inference import L2ArcticInference
from phoneme_assessment.alignment import PronunciationAligner

class PronunciationAssessmentSystem:
    def __init__(self, model_dir="d:/test/wav2vec2-l2arctic_final"):
        """
        Khởi tạo Hệ thống Đánh giá Phát âm toàn diện.
        Kết hợp 2 khối:
        1. Khối Nhận diện Âm thanh của AI (Inference)
        2. Khối Dịch Text & Phân tích Gióng hàng Lỗi (Aligner)
        """
        print(f"🔧 Đang khởi động Hệ thống Đánh giá Phát âm...")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục mô hình tại {model_dir}")
            
        # Nạp bộ công cụ
        self.ai = L2ArcticInference(model_dir)
        self.aligner = PronunciationAligner()
        print(f"✅ Hệ thống đã sẵn sàng!\n")
        
    def assess_audio(self, audio_path, reference_text):
        """
        Xử lý từ A-Z một file audio theo chuẩn bài toán L2-ARCTIC.
        """
        print(f"-"*50)
        print(f"🎤 File Audio: {audio_path}")
        print(f"📝 Text cần đọc: '{reference_text}'")
        print(f"-"*50)

        # 1. Dịch Text chuẩn sang mảng Phoneme chuẩn
        ref_phonemes = self.aligner.text_to_phonemes(reference_text)
        
        # 2. Nghe âm thanh thực tế của người dùng và nhận diện Phoneme
        pred_phonemes = self.ai.predict(audio_path)
        
        # In ra 2 mảng để debug
        print(f"    [CHUẨN G2P] : {ref_phonemes}")
        print(f"    [AI NGHE ĐƯỢC]  : {pred_phonemes}")
        print("\n📊 BÁO CÁO PHÂN TÍCH LỖI:")
        
        # 3. Chấm các lỗi (Substitution, Deletion, Insertion)
        results = self.aligner.align_and_grade(ref_phonemes, pred_phonemes)
        
        # Thống kê
        err_count = 0
        total_phonemes = len(ref_phonemes)
        
        for r in results:
            if r['type'] == 'correct':
                print(f"    ✅ Cần: {r['expected']:4} -> Thực tế: {r['actual']} (Chuẩn)")
            else:
                err_count += 1
                color = ""
                # Định dạng các loại mã màu để dễ nhìn trên console (nếu hỗ trợ)
                if r['type'] == 'substitution': 
                    color = "❌ [ĐỌC SAI]"
                elif r['type'] == 'deletion': 
                    color = "⚠️ [MẤT ÂM]"
                elif r['type'] == 'insertion': 
                    color = "🔴 [THỪA ÂM]"
                    
                print(f"    {color:15} Cần: {r['expected']:4} -> Thực tế: {r['actual']}")
                
        # Tính điểm
        if total_phonemes > 0:
            score = max(0, 100 - (err_count / total_phonemes * 100))
        else:
            score = 0
            
        print(f"\n🎯 ĐIỂM CHUẨN XÁC: {score:.2f} / 100")
        print(f"    (Tổng số âm vị: {total_phonemes} | Số lỗi: {err_count})")
        print(f"-"*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hệ Thống Đánh Giá Phát Âm Tiếng Anh (L2-ARCTIC)")
    parser.add_argument("--audio", type=str, required=False, default=r"d:\test\l2arctic_release_v5.0\ABA\wav\arctic_a0001.wav", help="Đường dẫn đến file Audio (.wav)")
    parser.add_argument("--text", type=str, required=False, default="Author of the danger trail, Philip Steels", help="Văn bản tiếng Anh cần đọc")
    parser.add_argument("--model_dir", type=str, default=r"d:\test\wav2vec2-l2arctic_final", help="Đường dẫn chứa mô hình AI đã train")
    
    args = parser.parse_args()
    
    try:
        app = PronunciationAssessmentSystem(model_dir=args.model_dir)
        
        if os.path.exists(args.audio):
            app.assess_audio(audio_path=args.audio, reference_text=args.text)
        else:
            print(f"Không tìm thấy file Audio: {args.audio}. Bạn hãy gửi đường dẫn chính xác (ví dụ: c:/my_audio.wav)")
            
    except Exception as e:
        print(f"LỖI: {str(e)}")
