import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv()


class Settings:
    """Application configuration (minimal)

    Values are intentionally simple and safe for local development.
    """

    FIREBASE_CREDS_PATH: str = os.getenv('FIREBASE_CREDS_PATH', 'firebase-adminsdk.json')
    FIREBASE_DATABASE_URL: Optional[str] = os.getenv('FIREBASE_DATABASE_URL')
    FIREBASE_STORAGE_BUCKET: Optional[str] = os.getenv('FIREBASE_STORAGE_BUCKET')

    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD') or None

    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.85'))
    ANPR_CONFIDENCE_THRESHOLD: float = float(os.getenv('ANPR_CONFIDENCE_THRESHOLD', '0.75'))

    VIDEO_SOURCE: str = os.getenv('VIDEO_SOURCE', '0')

    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def validate(cls) -> bool:
        # minimal validation
        if not Path(cls.FIREBASE_CREDS_PATH).exists():
            # not an error for this scaffold; return False to indicate missing creds
            return False
        return True


def get_settings() -> Settings:
    return Settings()
