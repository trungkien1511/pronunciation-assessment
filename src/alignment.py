import re
from g2p_en import G2p
import Levenshtein

class PronunciationAligner:
    def __init__(self):
        """Khởi tạo Dictionary Dịch Grapheme-to-Phoneme của tiếng Anh chuẩn."""
        self.g2p = G2p()
        
    def text_to_phonemes(self, sentence_text):
        """
        Bước 2: Dịch một câu Text tiếng Anh bình thường (Grapheme) 
        sang chuỗi Mẫu Phiên Âm Chuẩn (Reference Phonemes).
        """
        # 1. Dùng g2p_en để dịch (thường trả về cả dấu ngắt, số 0, 1, 2 đánh dấu trọng âm)
        raw_phonemes = self.g2p(sentence_text)
        
        # 2. Xóa các ký tự thừa (khoảng trắng, dấu câu chữ)
        # Chỉ giữ lại hệ thống Arpabet chuẩn như AH, T, D. Loại bỏ số đằng sau.
        clean_phonemes = []
        for p in raw_phonemes:
            # Bỏ dấu cách và dấu câu
            if p.strip() in ['', '.', ',', '?', '!', ':', ';']:
                continue
            # Xóa số 0, 1, 2 đánh dấu trọng âm (ví dụ: AH0 -> AH)
            clean_p = re.sub(r'\d', '', p)
            clean_phonemes.append(clean_p)
            
        return clean_phonemes

    def align_and_grade(self, reference_phonemes, predicted_phonemes):
        """
        Bước 3: Thuật toán Gióng Hàng (Sequence Alignment) giữa Chuẩn và Máy Nghe Được.
        Sử dụng Levenshtein Editops để tìm chính xác lỗi ở vị trí nào.
        """
        
        # Chúng ta gán cho mỗi Phoneme 1 ID tạm thời dưới dạng ký tự (chr) 
        # Vì hàm Levenshtein.editops thiết kế tối ưu cực tốt cho chuỗi String.
        # Ví dụ: ['W', 'AH', 'T'] -> 'ABC'
        
        vocab_list = list(set(reference_phonemes + predicted_phonemes))
        dict_to_char = {ph: chr(i + 65) for i, ph in enumerate(vocab_list)}
        
        # Đổi mảng Phoneme thành String (ký tự tự chế quy ước)
        ref_str = "".join([dict_to_char[p] for p in reference_phonemes])
        pred_str = "".join([dict_to_char[p] for p in predicted_phonemes])
        
        # 🚀 TÍNH TOÁN KHOẢNG CÁCH VÀ VỊ TRÍ SAI LỆCH 
        # Kết quả sẽ ra dạng: [('replace', 1, 1), ('insert', 3, 3)...]
        edits = Levenshtein.editops(ref_str, pred_str)
        
        # 📊 ĐÓNG GÓI KẾT QUẢ ĐÁNH GIÁ TỪNG TỪ
        report = []
        ref_idx = 0
        pred_idx = 0
        edit_idx = 0
        
        while ref_idx < len(reference_phonemes) or pred_idx < len(predicted_phonemes):
            # Nếu tại vị trí này có lỗi
            if edit_idx < len(edits):
                op, r_pos, p_pos = edits[edit_idx]
                
                # 1. Lỗi Xóa/Thiếu âm (Deletion) - Người nói nuốt âm
                if op == 'delete' and ref_idx == r_pos:
                    report.append({
                        "type": "deletion",
                        "expected": reference_phonemes[r_pos],
                        "actual": "∅ (Mất âm)"
                    })
                    ref_idx += 1
                    edit_idx += 1
                    continue
                    
                # 2. Lỗi Chèn âm thừa (Insertion) - Người nói bị nhịu, đọc thêm âm lạ
                elif op == 'insert' and pred_idx == p_pos:
                    report.append({
                        "type": "insertion",
                        "expected": "∅ (Không có)",
                        "actual": predicted_phonemes[p_pos]
                    })
                    pred_idx += 1
                    edit_idx += 1
                    continue
                    
                # 3. Lỗi Đọc sai âm (Substitution) - Ngọng
                elif op == 'replace' and ref_idx == r_pos and pred_idx == p_pos:
                    report.append({
                        "type": "substitution",
                        "expected": reference_phonemes[r_pos],
                        "actual": predicted_phonemes[p_pos]
                    })
                    ref_idx += 1
                    pred_idx += 1
                    edit_idx += 1
                    continue

            # 4. Khi r_pos r_idx bằng nhau mà khooog rớt vào if (Tức Là LÀm Đúng)
            if ref_idx < len(reference_phonemes) and pred_idx < len(predicted_phonemes):
                 report.append({
                    "type": "correct",
                    "expected": reference_phonemes[ref_idx],
                    "actual": predicted_phonemes[pred_idx]
                 })
                 ref_idx += 1
                 pred_idx += 1
                 
            # Xử lý edge cases độ dài không đều ở đuôi
            elif ref_idx < len(reference_phonemes):
                report.append({
                    "type": "deletion",
                    "expected": reference_phonemes[ref_idx],
                    "actual": "∅ (Mất âm)"
                })
                ref_idx += 1
            else:
                report.append({
                    "type": "insertion",
                    "expected": "∅",
                    "actual": predicted_phonemes[pred_idx]
                })
                pred_idx += 1

        return report

if __name__ == "__main__":
    print("Test hệ thống Dịch Text và Gióng hàng Lỗi...")
    aligner = PronunciationAligner()
    
    # 1. Text chuẩn đầu vào
    text = "Hello world"
    ref_ph = aligner.text_to_phonemes(text)
    print(f"Bản chuẩn G2P [{text}]:", ref_ph)
    
    # 2. Giả lập kết quả mô hình AI nghe bị lỗi 
    # Hello world -> HH AH L OW W ER L D (Chuẩn)
    # Máy nghe ra  -> HH EH N OW W ER D   (Ngọng chữ L thành N, nuốt chữ L thứ hai)
    pred_ph = ['HH', 'EH', 'N', 'OW', 'W', 'ER', 'D']
    print(f"Bản AI nghe được:", pred_ph)
    
    # 3. Chấm điểm
    print("\n[BÁO CÁO PHÁT ÂM CHI TIẾT]")
    results = aligner.align_and_grade(ref_ph, pred_ph)
    for r in results:
        status = "✅" if r['type'] == 'correct' else "❌"
        print(f"{status} {r['type'].upper():13} | Cần đọc: {r['expected']:4} | Thực tế: {r['actual']}")
