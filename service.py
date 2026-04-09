import os
from .core.inference import L2ArcticInference
from .core.alignment import PronunciationAligner
from .config import settings

# Bảng chuyển đổi ARPAbet → IPA (Phiên Âm Quốc Tế) cho response dễ đọc
ARPABET_TO_IPA = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AX": "ə", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɜːr", "EY": "eɪ",
    "F": "f", "G": "ɡ", "HH": "h",
    "IH": "ɪ", "IY": "iː",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ",
    "P": "p", "R": "r", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ",
    "UH": "ʊ", "UW": "uː",
    "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"
}

def to_ipa(phoneme: str) -> str:
    if phoneme is None:
        return None
    return ARPABET_TO_IPA.get(phoneme, phoneme)

class PronunciationAssessmentService:
    def __init__(self):
        self.ai_engine = None
        self.aligner = None
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            return True
            
        print(f"🔧 [Pronunciation Module] Đang nạp mô hình AI từ: {settings.DEFAULT_MODEL_DIR}")
        if not os.path.exists(settings.DEFAULT_MODEL_DIR):
            print(f"⚠️ CẢNH BÁO: Không tìm thấy thư mục model '{settings.DEFAULT_MODEL_DIR}'. Module Pronunciation sẽ không hoạt động.")
            return False
            
        self.ai_engine = L2ArcticInference(settings.DEFAULT_MODEL_DIR, settings.VOCAB_PATH)
        self.aligner = PronunciationAligner()
        self.is_loaded = True
        print("✅ [Pronunciation Module] Khởi tạo thành công!")
        return True

    def process_audio(self, audio_filepath: str, reference_text: str):
        """Logic chính nhận file audio và chữ, trả về kết quả JSON chuẩn hoá"""
        if not self.is_loaded:
            raise Exception("Model chưa được nạp vào bộ nhớ.")
            
        # 1. AI nghe âm thanh -> chuỗi phoneme thực tế
        pred_phonemes = self.ai_engine.predict(audio_filepath)
        
        # 2. Alignment & Đánh giá theo từng từ (tính cả âm vị lỗi)
        word_info, all_ref = self.aligner.assess_by_words(reference_text, pred_phonemes)
        
        # 3. Tính điểm tổng
        total_phonemes = len(all_ref)
        total_errors = 0
        
        word_details = []
        for wi in word_info:
            char_map = wi["char_map"]   
            results = wi["results"]     
            
            phoneme_entries = []
            cm_idx = 0  

            for r in results:
                if r["type"] == "insertion":
                    actual_raw = r["actual"]
                    if actual_raw and "∅" not in actual_raw:
                        phoneme_entries.append({
                            "char": None,
                            "expected": None,
                            "actual": to_ipa(actual_raw),
                            "type": "insertion"
                        })
                else:
                    chars = char_map[cm_idx][0] if cm_idx < len(char_map) else ""
                    expected_raw = r["expected"]
                    actual_raw = r["actual"]
                    
                    phoneme_entries.append({
                        "char": chars,
                        "expected": to_ipa(expected_raw) if expected_raw and "∅" not in expected_raw else None,
                        "actual": to_ipa(actual_raw) if actual_raw and "∅" not in actual_raw else None,
                        "type": r["type"]
                    })
                    cm_idx += 1

            # Gộp lại theo logic api cũ (Merge silences / syllabic endings)
            merged_entries = []
            for entry in phoneme_entries:
                if (not entry.get("char")) and entry["type"] != "insertion" and merged_entries:
                    prev = merged_entries[-1]
                    prev["expected"] = entry["expected"]
                    if entry["type"] not in ["correct", "soft_correct"]:
                        prev["type"] = entry["type"]
                        prev["actual"] = entry["actual"]
                    elif prev["type"] in ["correct", "soft_correct"]:
                        prev["actual"] = entry["actual"]
                else:
                    merged_entries.append(entry)
            phoneme_entries = merged_entries

            # Đếm lỗi (soft_correct đếm 0.2)
            word_errors = sum(
                0 if e["type"] == "correct" else (0.2 if e["type"] == "soft_correct" else 1.0)
                for e in phoneme_entries
            )
            total_errors += word_errors
            word_score = max(0, 100 - (word_errors / max(len(phoneme_entries), 1) * 100))

            word_details.append({
                "word": wi["word"],
                "score": round(word_score, 2),
                "phonemes": phoneme_entries
            })

        overall_score = max(0, 100 - (total_errors / total_phonemes * 100)) if total_phonemes > 0 else 0
        per = (total_errors / total_phonemes * 100) if total_phonemes > 0 else 0

        return {
            "overall_score": round(overall_score, 2),
            "phoneme_error_rate": round(per, 2),
            "total_phonemes": total_phonemes,
            "error_count": total_errors,
            "word_details": word_details
        }

# Khởi tạo instance global
pronunciation_service = PronunciationAssessmentService()
