# 🔌 API Specification: Đánh Giá Phát Âm (Dành Cho FE)
Tài liệu định dạng chuẩn (như Swagger/Postman) để Frontend mapping và tích hợp trực tiếp, không chứa mã mẫu.

---

## `[POST] /api/v1/assess_pronunciation`
**Chức năng:** Gửi tệp âm thanh và văn bản mẫu để nhận về mảng điểm số độ chính xác phát âm.

### 1. Headers Request
| Thuộc tính | Bắt buộc | Giá trị |
|---|:---:|---|
| `Content-Type` | ✅ | `multipart/form-data` |
| `Authorization` | ✅ | `Bearer <JWT_TOKEN>` |

### 2. Body Request (Form-Data)
| Biến (Field) | Loại (Type) | Bắt buộc | Ràng buộc |
|---|---|:---:|---|
| `audio_file` | `File` | ✅ | Định dạng `.wav`, `.mp3`. Tối đa 10MB. |
| `reference_text` | `String`| ✅ | Độ dài văn bản mẫu < 1000 ký tự. |

---

### 3. Response: Thành Công (HTTP 200)
Trả về danh sách Array ánh xạ 1-1 từng âm vị, kèm theo loại lỗi và mã màu HEX gợi ý cho Frontend.

```json
{
  "status": "success",
  "data": {
    "overall_score": 50.0,
    "phoneme_error_rate": 50.0,
    "alignment_details": [
      {
        "expected": "HH",
        "actual": "HH",
        "type": "correct",
        "color_code": "#00FF00"
      },
      {
        "expected": "EH",
        "actual": "AH",
        "type": "substitution",
        "color_code": "#FF0000"
      },
      {
        "expected": "L",
        "actual": null,
        "type": "deletion",
        "color_code": "#FFA500"
      },
      {
        "expected": null,
        "actual": "S",
        "type": "insertion",
        "color_code": "#FFA500"
      }
    ]
  }
}
```

> **Ghi chú giá trị mảng `type` cho FE xử lý:**
> *   `correct`: Đọc đúng âm.
> *   `substitution`: Phát âm sai (Thay thế bằng âm khác).
> *   `deletion`: Bị lướt/nuốt âm (Không phát ra).
> *   `insertion`: Đọc thừa, bị nhịu thêm một âm không có trong bảng chữ cái.

---

### 4. Response: Thất Bại (HTTP Lỗi)

**HTTP 401: Lỗi Token / Chưa Đăng Nhập**
```json
{
  "status": "error",
  "error_code": "UNAUTHORIZED",
  "message": "Vui lòng đính kèm JWT Token hợp lệ vào header."
}
```

**HTTP 400: Lỗi Dữ Liệu Thiếu/Sai Định Dạng**
```json
{
  "status": "error",
  "error_code": "INVALID_FILE_FORMAT",
  "message": "Chỉ nhận định dạng file wav hoặc mp3."
}
```

**HTTP 500: Server bận/Lỗi mô hình**
```json
{
  "status": "error",
  "error_code": "INFERENCE_ERROR",
  "message": "Không thể xử lý âm thanh lúc này."
}
```
