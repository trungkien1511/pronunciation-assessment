import os

class Settings:
    # Lấy thư mục gốc (nơi chứa code này)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # API & Model Paths
    DEFAULT_MODEL_DIR = os.getenv("PRONUNCIATION_MODEL_DIR", os.path.join(BASE_DIR, "models", "wav2vec2-l2arctic_finetuned_v3"))
    VOCAB_PATH = os.path.join(BASE_DIR, "data", "vocab.json")
    
    # Lưu file từ client (lưu ở thư mục tạm hệ thống hoặc debug)
    DEBUG_AUDIO_DIR = os.path.join(BASE_DIR, "debug_audio_from_client")
    
    # Validate configurations
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))
    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}

settings = Settings()

os.makedirs(settings.DEBUG_AUDIO_DIR, exist_ok=True)
