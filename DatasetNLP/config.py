"""
Configuration settings for NLP dataset generation.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_CACHE_DIR = BASE_DIR / ".model_cache"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Text settings
MAX_TEXT_LENGTH = 512
BATCH_SIZE = 32

# Supported languages
SUPPORTED_LANGUAGES = ["EN", "TH", "JA", "KO", "ZH", "VI"]
DEFAULT_LANGUAGE = "EN"

# Supported NLP tasks
TASK_CONFIG = {
    "text_classification": {
        "name": "Text Classification",
        "description": "Classify text into predefined categories",
        "models": [
            "bert-base-multilingual-cased",
            "xlm-roberta-base",
            "facebook/muppet-roberta-base"
        ],
        "parameters": {
            "top_k": {
                "name": "Top K Predictions",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 5
            },
            "batch_size": {
                "name": "Batch Size",
                "type": "number",
                "min": 1,
                "max": 64,
                "default": 32
            }
        }
    },
    "token_classification": {
        "name": "Token Classification (NER)",
        "description": "Identify and classify named entities in text",
        "models": [
            "dslim/bert-base-NER-uncased",
            "xlm-roberta-large-finetuned-conll03-english",
            "elastic/distilbert-base-uncased-finetuned-conll03-english"
        ],
        "parameters": {
            "confidence_threshold": {
                "name": "Confidence Threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5
            },
            "batch_size": {
                "name": "Batch Size",
                "type": "number",
                "min": 1,
                "max": 64,
                "default": 32
            }
        }
    },
    "text_generation": {
        "name": "Text Generation",
        "description": "Generate text continuations",
        "models": [
            "gpt2",
            "EleutherAI/gpt-neo-125M",
            "facebook/opt-125m"
        ],
        "parameters": {
            "max_new_tokens": {
                "name": "Maximum New Tokens",
                "type": "number",
                "min": 1,
                "max": 200,
                "default": 50
            },
            "temperature": {
                "name": "Temperature",
                "type": "number",
                "min": 0.1,
                "max": 2.0,
                "default": 0.7
            },
            "include_prompt": {
                "name": "Include Prompt in Output",
                "type": "boolean",
                "default": True
            }
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    "use_fp16": os.environ.get("USE_FP16", "1") == "1",
    "device": "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu",
    "num_workers": int(os.environ.get("NUM_WORKERS", "4")),
    "pin_memory": True,
    "cache_dir": MODEL_CACHE_DIR
}

# Cache settings
ENABLE_CACHE = True
CACHE_DIR = BASE_DIR / ".cache"
CACHE_TTL = 24 * 60 * 60  # 24 hours
MAX_CACHE_SIZE = 1000  # entries

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_FILENAME = "nlp_dataset.log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Create additional directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
VISUALIZATION = {
    "max_display_examples": 5,
    "token_colors": {
        "PER": "#7aecec",
        "ORG": "#bfeeb7",
        "LOC": "#feca74",
        "MISC": "#ff9561"
    }
}