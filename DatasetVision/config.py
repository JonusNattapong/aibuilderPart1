"""
Configuration settings for vision dataset generation.
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

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Supported vision tasks
SUPPORTED_TASKS = {
    "image_classification": {
        "name": "Image Classification",
        "parameters": {
            "top_k": {
                "name": "Top K Predictions",
                "type": "number",
                "min": 1,
                "max": 10,
                "default": 5
            }
        }
    },
    "object_detection": {
        "name": "Object Detection",
        "parameters": {
            "confidence_threshold": {
                "name": "Confidence Threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5
            }
        }
    },
    "image_segmentation": {
        "name": "Image Segmentation",
        "parameters": {
            "mask_threshold": {
                "name": "Mask Threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5
            }
        }
    },
    "depth_estimation": {
        "name": "Depth Estimation",
        "parameters": {
            "min_depth": {
                "name": "Minimum Depth",
                "type": "number",
                "min": 0.1,
                "max": 10.0,
                "default": 1.0
            }
        }
    },
    "keypoint_detection": {
        "name": "Keypoint Detection",
        "parameters": {
            "keypoint_threshold": {
                "name": "Keypoint Confidence Threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "default": 0.3
            }
        }
    }
}

# Default models for each task
DEFAULT_MODEL = "microsoft/resnet-50"

SUPPORTED_MODELS = {
    "image_classification": [
        "microsoft/resnet-50",
        "google/vit-base-patch16-224",
        "facebook/deit-base-distilled-patch16-224"
    ],
    "object_detection": [
        "facebook/detr-resnet-50",
        "microsoft/faster-rcnn-resnet-50-fpn"
    ],
    "image_segmentation": [
        "facebook/mask2former-swin-large-coco-instance",
        "nvidia/segformer-b0-finetuned-ade-512-512"
    ],
    "depth_estimation": [
        "intel/dpt-large",
        "facebook/dino-vitb16"
    ],
    "keypoint_detection": [
        "microsoft/hrnet-w32",
        "apple/mobilenet-v3-large"
    ]
}

# Model configuration
MODEL_CONFIG = {
    "use_fp16": True,  # Use mixed precision
    "device": "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu",
    "num_workers": int(os.environ.get("NUM_WORKERS", "4")),
    "pin_memory": True
}

# Cache settings
ENABLE_CACHE = True
CACHE_DIR = BASE_DIR / ".cache"
CACHE_TTL = 24 * 60 * 60  # 24 hours
MAX_CACHE_SIZE = 1000  # entries

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_FILENAME = "vision_dataset.log"
LOG_LEVEL = "INFO"

# Create additional directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Export formats
SUPPORTED_FORMATS = ["JSON", "CSV", "JSONL"]
DEFAULT_FORMAT = "JSONL"

# Visualization settings
VISUALIZATION = {
    "max_display_images": 5,
    "bbox_color": (255, 0, 0),  # Red
    "keypoint_color": (0, 255, 0),  # Green
    "mask_alpha": 0.5,
    "depth_colormap": "plasma"
}