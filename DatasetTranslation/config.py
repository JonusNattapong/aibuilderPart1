"""
Shared configuration settings for both API and Streamlit interfaces.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# DeepL API Settings
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "DataOutput" / "translations"
CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", ".cache")
LOG_DIR = BASE_DIR / "logs"

# File names
LOG_FILENAME = "translation.log"
CACHE_FILENAME = "translation_cache.db"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Translation Settings
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
VALIDATION_SETTINGS = {
    "length_ratio_min": 0.5,
    "length_ratio_max": 2.0,
    "require_language_match": True
}

# Output Formats
SUPPORTED_FORMATS = ["CSV", "JSON", "JSONL"]
DEFAULT_FORMAT = "JSONL"

# Create required directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# File type settings
SUPPORTED_FILE_TYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/jsonl"
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Cache settings
CACHE_TTL = 24 * 60 * 60  # 24 hours
MAX_CACHE_SIZE = 1000  # entries