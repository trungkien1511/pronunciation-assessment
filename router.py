from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
from datetime import datetime

from .config import settings
from .service import pronunciation_service

# =====================================================================
# Tạo APIRouter thay vì FastAPI().
# Đây là cách để "cắt" module này gắn vào 1 server có 100 model khác
# Server gốc chỉ cần gọi: app.include_router(pronunciation_router)
# =====================================================================
router = APIRouter(
    prefix="/api/v1/pronunciation",
    tags=["Pronunciation Assessment"],
    responses={404: {"description": "Not found"}},
)


@router.post("/assess")
async def assess_pronunciation_endpoint(
    audio_file: UploadFile = File(..., description="File ghi âm (.wav, .mp3)"),
    reference_text: str = Form(..., description="Câu tiếng Anh cần đọc")
):
    if not pronunciation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Mô hình Pronunciation AI chưa sẵn sàng.")

    # Validate file định dạng & kích thước
    file_ext = os.path.splitext(audio_file.filename or "")[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Chỉ hỗ trợ {settings.ALLOWED_EXTENSIONS}"})

    contents = await audio_file.read()
    if len(contents) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        return JSONResponse(status_code=400, content={"status": "error", "message": "File quá cỡ."})

    # Lưu file để AI engine đọc
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = audio_file.filename.replace(" ", "_") if audio_file.filename else "audio.wav"
    saved_filepath = os.path.join(settings.DEBUG_AUDIO_DIR, f"{timestamp}_{safe_name}")
    
    with open(saved_filepath, "wb") as f:
        f.write(contents)

    # Chạy qua Service của module
    try:
        result_payload = pronunciation_service.process_audio(saved_filepath, reference_text)
        
        return {
            "status": "success",
            "data": result_payload
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.get("/health")
def health_check():
    return {
        "status": "ok", 
        "module": "Pronunciation Assessment",
        "model_loaded": pronunciation_service.is_loaded,
        "default_model": settings.DEFAULT_MODEL_DIR
    }
