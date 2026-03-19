import os
import re
import textgrid

def clean_phoneme(phoneme):
    """
    Làm sạch 1 phoneme: Loại bỏ stress markers (số 0,1,2) và các ký tự đặc biệt.
    Ví dụ: 'AH0' -> 'AH', 'EY1' -> 'EY', 'V``' -> 'V'
    """
    # Loại bỏ các ký tự đặc biệt (backtick, underscore ở cuối)
    phoneme = phoneme.rstrip('`_')
    # Loại bỏ số 0, 1, 2 đánh dấu trọng âm (stress markers)
    phoneme = re.sub(r'\d+$', '', phoneme)
    return phoneme

# Bộ Phoneme ARPAbet chuẩn (chỉ chấp nhận những phoneme trong danh sách này)
VALID_ARPABET = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY',
    'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY',
    'F', 'G', 'HH',
    'IH', 'IY',
    'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW', 'OY',
    'P', 'R', 'S', 'SH',
    'T', 'TH',
    'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH'
}

def is_valid_phoneme(phoneme):
    """Kiểm tra xem phoneme có nằm trong ARPAbet chuẩn không."""
    return phoneme in VALID_ARPABET

# Danh sách các nhãn KHÔNG PHẢI là phoneme thực tế, cần loại bỏ
NOISE_TOKENS = {'sil', 'sp', 'spn', '', '{SL}', '{LG}', '{CG}', '{BR}', '{NS}'}

def parse_textgrid(file_path):
    """
    Phân tích file TextGrid từ L2-ARCTIC và trích xuất danh sách âm vị chuẩn cùng nhãn lỗi.
    """
    try:
        tg = textgrid.TextGrid.fromFile(file_path)
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return [], []
        
    # Lấy tier 'phones'
    phones_tier = tg.getFirst('phones')
    if phones_tier is None:
        print(f"Không tìm thấy tier 'phones' trong file {file_path}")
        return [], []
        
    reference_phonemes = []
    labels = []
    
    for interval in phones_tier:
        mark = interval.mark.strip()
        
        # Bỏ qua khoảng im lặng và tạp âm
        if mark.lower() in NOISE_TOKENS or mark == '':
            continue
            
        # Tách chuỗi nhãn dựa trên dấu phẩy
        parts = [p.strip() for p in mark.split(',')]
        
        if len(parts) == 3:
            cpl, ppl, err_type = parts
            
            # Làm sạch phoneme chuẩn (CPL)
            cpl_clean = clean_phoneme(cpl)
            
            # Bỏ qua nếu CPL sau khi clean không phải phoneme hợp lệ trong ARPAbet
            if not is_valid_phoneme(cpl_clean):
                continue
            
            if err_type == 's':
                # Substitution
                reference_phonemes.append(cpl_clean)
                labels.append('substitution')
            elif err_type == 'd':
                # Deletion
                reference_phonemes.append(cpl_clean)
                labels.append('deletion')
            elif err_type == 'a':
                # Insertion: Bỏ qua hoàn toàn vì reference không có âm này
                # (Đó là âm thừa do user thêm vào, reference gốc không có)
                continue
            else:
                # Loại lỗi không xác định → bỏ qua
                continue
        
        elif len(parts) == 1:
            # Phoneme đơn = Phát âm đúng
            clean = clean_phoneme(mark)
            
            # Kiểm tra xem phoneme có hợp lệ trong danh sách ARPAbet
            if not is_valid_phoneme(clean):
                continue
                
            reference_phonemes.append(clean)
            labels.append('correct')
            
        else:
            # Định dạng không hợp lệ → bỏ qua
            continue

    return reference_phonemes, labels

# Test
if __name__ == "__main__":
    test_path = "sample.TextGrid" 
    if os.path.exists(test_path):
        refs, states = parse_textgrid(test_path)
        print("Mảng Phoneme chuẩn:\n", refs)
        print("Mảng Label trạng thái:\n", states)
    else:
        print(f"File test {test_path} không tồn tại.")
