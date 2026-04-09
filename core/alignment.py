import re
from g2p_en import G2p
import Levenshtein

# =====================================================================
#  BẢNG ÁNH XẠ KÝ TỰ → ÂM VỊ (Grapheme-Phoneme Mapping)
# =====================================================================
CHAR_TO_PHONEMES = {
    'a': {'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'EY'},
    'b': {'B'},
    'c': {'K', 'S', 'CH', 'SH'},
    'd': {'D', 'JH'},
    'e': {'AH', 'EH', 'ER', 'EY', 'IH', 'IY'},
    'f': {'F'},
    'g': {'G', 'JH'},
    'h': {'HH'},
    'i': {'AH', 'AY', 'IH', 'IY'},
    'j': {'JH', 'Y'},
    'k': {'K'},
    'l': {'L', 'AH'},         # AH cho syllabic L (apple, bottle, little)
    'm': {'M'},
    'n': {'N', 'NG', 'AH'},   # AH cho syllabic N (button, kitten)
    'o': {'AA', 'AH', 'AO', 'AW', 'OW', 'OY', 'UH', 'UW'},
    'p': {'P'},
    'q': {'K'},
    'r': {'R', 'ER', 'AH'},  # AH cho syllabic R
    's': {'S', 'SH', 'Z', 'ZH'},
    't': {'T', 'CH', 'SH'},
    'u': {'AH', 'UH', 'UW', 'W', 'Y', 'ER'},
    'v': {'V'},
    'w': {'W'},
    'x': {'K', 'S', 'Z'},
    'y': {'AY', 'IH', 'IY', 'Y'},
    'z': {'S', 'Z', 'ZH'},
}

DIGRAPH_TO_PHONEMES = {
    'ch': {'CH', 'K', 'SH'},
    'ck': {'K'},
    'dg': {'JH'},
    'gh': {'F', 'G'},
    'gn': {'N'},
    'kn': {'N'},
    'ph': {'F'},
    'sh': {'SH'},
    'th': {'DH', 'TH'},
    'wh': {'HH', 'W'},
    'wr': {'R'},
    # Phụ âm đôi (doubled consonants) → 1 âm duy nhất
    'bb': {'B'},
    'cc': {'K', 'S'},
    'dd': {'D'},
    'ff': {'F'},
    'gg': {'G', 'JH'},
    'll': {'L'},
    'mm': {'M'},
    'nn': {'N'},
    'pp': {'P'},
    'rr': {'R'},
    'ss': {'S', 'Z'},
    'tt': {'T'},
    'zz': {'Z'},
    # Nguyên âm đôi (vowel digraphs) → 1 âm
    'ea': {'IY', 'EH', 'EY', 'ER'},
    'ee': {'IY'},
    'oo': {'UW', 'UH'},
    'ou': {'AW', 'AH', 'UW', 'OW'},
    'ow': {'OW', 'AW'},
    'oi': {'OY'},
    'oy': {'OY'},
    'ai': {'EY', 'EH'},
    'ay': {'EY'},
    'au': {'AO', 'AA'},
    'aw': {'AO'},
    'ew': {'UW', 'Y'},
    'ei': {'IY', 'EY', 'AY'},
    'ie': {'IY', 'AY', 'IH'},
    'ue': {'UW'},
    'ge': {'JH'},
}

# =====================================================================
#  BẢNG CÁC CẶP ÂM TƯƠNG ĐỒNG (Phonetic Similarity / Confusable Pairs)
#  Giúp xử lý các lỗi nhầm lẫn phổ biến của AI khi người dùng đọc nhanh
# =====================================================================
PHONETIC_SIMILARITY = {
    ('G', 'K'), ('K', 'G'),   # Velar Stops (Voiced/Voiceless)
    ('D', 'T'), ('T', 'D'),   # Alveolar Stops
    ('B', 'P'), ('P', 'B'),   # Bilabial Stops
    ('V', 'F'), ('F', 'V'),   # Labiodental Fricatives
    ('Z', 'S'), ('S', 'Z'),   # Alveolar Fricatives
    ('JH', 'CH'), ('CH', 'JH'), # Affricates
    ('DH', 'TH'), ('TH', 'DH'), # Dental Fricatives
    ('ZH', 'SH'), ('SH', 'ZH'), # Post-alveolar Fricatives
    ('M', 'N'), ('N', 'M'),     # Nasals
    ('AH', 'AX'), ('AX', 'AH'), # Schwa-like vowels
    ('IH', 'IY'), ('IY', 'IH'), # High front vowels (short/long)
    ('UH', 'UW'), ('UW', 'UH'), # High back vowels (short/long)
}


def align_chars_phonemes(word, phonemes):
    """
    Gióng hàng tham lam (greedy) từ trái sang phải giữa các ký tự trong từ và âm vị ARPAbet.
    Trả về danh sách các tuple (ký_tự, âm_vị).

    Ví dụ: align_chars_phonemes("grape", ["G", "R", "EY", "P"])
           → [("g", "G"), ("r", "R"), ("a", "EY"), ("pe", "P")]
    """
    w = word.lower()
    result = []
    ci = 0  # character index
    pi = 0  # phoneme index

    while pi < len(phonemes) and ci < len(w):
        ph = phonemes[pi]
        matched = False

        # 1. Thử digraph (2 ký tự → 1 âm vị): sh→SH, th→TH, ph→F...
        if ci + 1 < len(w):
            di = w[ci:ci + 2]
            if di in DIGRAPH_TO_PHONEMES and ph in DIGRAPH_TO_PHONEMES[di]:
                result.append((word[ci:ci + 2], ph))
                ci += 2
                pi += 1
                matched = True

        # 2. Thử 1 ký tự → 1 âm vị
        if not matched:
            ch = w[ci]
            if ch in CHAR_TO_PHONEMES and ph in CHAR_TO_PHONEMES[ch]:
                result.append((word[ci:ci + 1], ph))
                ci += 1
                pi += 1
                matched = True

        # 3. Ký tự câm (silent letter) → gộp vào entry trước đó
        if not matched:
            if result:
                prev_chars, prev_ph = result[-1]
                result[-1] = (prev_chars + word[ci], prev_ph)
            else:
                result.append((word[ci], None))  # buffer cho entry tiếp theo
            ci += 1

    # Các ký tự câm còn thừa ở cuối → gộp vào entry cuối
    while ci < len(w):
        if result:
            prev_chars, prev_ph = result[-1]
            result[-1] = (prev_chars + word[ci], prev_ph)
        ci += 1

    # Các phoneme còn thừa (hiếm khi xảy ra)
    while pi < len(phonemes):
        result.append(("", phonemes[pi]))
        pi += 1

    # Gộp các buffer None vào entry kế tiếp
    merged = []
    buf = ""
    for chars, ph in result:
        if ph is None:
            buf += chars
        else:
            merged.append((buf + chars, ph))
            buf = ""
    if buf and merged:
        c, p = merged[-1]
        merged[-1] = (c + buf, p)
    elif buf:
        merged.append((buf, ""))

    return merged


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
                    expected = reference_phonemes[r_pos]
                    actual = predicted_phonemes[p_pos]
                    
                    # Kiểm tra xem có phải cặp âm dễ nhầm lẫn không
                    is_similar = (expected, actual) in PHONETIC_SIMILARITY
                    
                    report.append({
                        "type": "soft_correct" if is_similar else "substitution",
                        "expected": expected,
                        "actual": actual,
                        "is_similar": is_similar
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

    def text_to_word_phonemes(self, sentence_text):
        """
        Dịch câu tiếng Anh sang phoneme theo TỪNG TỪ, kèm ánh xạ ký tự.
        Sử dụng ngữ cảnh câu (sentence-level G2P) để phiên âm chính xác hơn.

        Returns: list of {word, phonemes, char_map}
        """
        raw_phonemes = self.g2p(sentence_text)
        words = re.findall(r"[a-zA-Z']+", sentence_text)

        # Tách output G2P theo dấu cách (ranh giới từ)
        word_phoneme_groups = []
        current_group = []
        for p in raw_phonemes:
            if p == ' ':
                if current_group:
                    word_phoneme_groups.append(current_group)
                    current_group = []
            else:
                cleaned = p.strip()
                if cleaned and cleaned not in ['.', ',', '?', '!', ':', ';']:
                    clean_p = re.sub(r'\d', '', cleaned)
                    current_group.append(clean_p)
        if current_group:
            word_phoneme_groups.append(current_group)

        # Ghép từng nhóm phoneme vào từ tương ứng + ánh xạ ký tự
        result = []
        for i, word in enumerate(words):
            phs = word_phoneme_groups[i] if i < len(word_phoneme_groups) else []
            char_map = align_chars_phonemes(word, phs)
            result.append({
                "word": word,
                "phonemes": phs,
                "char_map": char_map,  # [(chars, phoneme), ...]
            })

        return result

    def assess_by_words(self, reference_text, predicted_phonemes):
        """
        Đánh giá phát âm theo TỪNG TỪ với ánh xạ ký tự.
        Kết hợp G2P word-level, Levenshtein alignment, và grapheme mapping.

        Returns: (word_details, all_ref_phonemes)
        """
        # 1. Lấy phoneme theo từ + ánh xạ ký tự
        word_info = self.text_to_word_phonemes(reference_text)

        # 2. Ghép tất cả phoneme chuẩn thành 1 mảng phẳng
        all_ref = []
        for wi in word_info:
            wi["ref_start"] = len(all_ref)
            all_ref.extend(wi["phonemes"])

        # 3. Gióng hàng toàn bộ chuỗi
        alignment = self.align_and_grade(all_ref, predicted_phonemes)

        # 4. Phân phối kết quả alignment về từng từ
        align_idx = 0
        for wi in word_info:
            wi["results"] = []
            consumed = 0
            target = len(wi["phonemes"])

            while align_idx < len(alignment) and consumed < target:
                entry = alignment[align_idx]
                wi["results"].append(entry)
                if entry["type"] != "insertion":
                    consumed += 1
                align_idx += 1

            # Gom các insertion còn dư trước từ kế tiếp
            while align_idx < len(alignment) and alignment[align_idx]["type"] == "insertion":
                wi["results"].append(alignment[align_idx])
                align_idx += 1

        return word_info, all_ref

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
