"""
Configuration settings for dataset generation.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Batch settings
BATCH_SIZE = 1000

# Supported formats
SUPPORTED_FORMATS = ["CSV", "PARQUET"]
DEFAULT_FORMAT = "CSV"

# CSV Configuration
CSV_CONFIG = {
    "tasks": {
        "qa_dataset": {
            "name": "Question-Answer Dataset",
            "schema": {
                "question": {"type": "string", "max_length": 200},
                "answer": {"type": "string", "max_length": 500},
                "category": {"type": "category", "categories": ["general", "technical", "business"]},
                "difficulty": {"type": "number", "min": 1, "max": 5},
                "timestamp": {"type": "datetime"}
            }
        },
        "user_profiles": {
            "name": "User Profiles Dataset",
            "schema": {
                "name": {"type": "string", "fake_type": "name"},
                "email": {"type": "string", "fake_type": "email"},
                "age": {"type": "number", "min": 18, "max": 90},
                "country": {"type": "string", "fake_type": "country"},
                "joined_date": {"type": "datetime"}
            }
        },
        "product_reviews": {
            "name": "Product Reviews Dataset",
            "schema": {
                "product_id": {"type": "string", "fake_type": "uuid4"},
                "user_id": {"type": "string", "fake_type": "uuid4"},
                "rating": {"type": "number", "min": 1, "max": 5},
                "review_text": {"type": "string", "max_length": 1000},
                "review_date": {"type": "datetime"}
            }
        }
    },
    "parameters": {
        "delimiter": {
            "name": "Delimiter",
            "type": "select",
            "options": [",", ";", "|", "\t"],
            "default": ","
        },
        "quoting": {
            "name": "Quote Characters",
            "type": "select",
            "options": ["minimal", "all", "none"],
            "default": "minimal"
        }
    }
}

# Parquet Configuration
PARQUET_CONFIG = {
    "tasks": {
        "sales_data": {
            "name": "Sales Transaction Dataset",
            "schema": {
                "transaction_id": {"type": "string", "fake_type": "uuid4"},
                "customer_id": {"type": "string", "fake_type": "uuid4"},
                "products": {"type": "list", "min_items": 1, "max_items": 10},
                "total_amount": {"type": "number", "min": 0, "max": 10000},
                "payment_method": {"type": "category", "categories": ["credit", "debit", "cash"]},
                "transaction_date": {"type": "datetime"}
            }
        },
        "sensor_data": {
            "name": "IoT Sensor Dataset",
            "schema": {
                "device_id": {"type": "string", "fake_type": "uuid4"},
                "temperature": {"type": "number", "min": -50, "max": 100},
                "humidity": {"type": "number", "min": 0, "max": 100},
                "pressure": {"type": "number", "min": 900, "max": 1100},
                "timestamp": {"type": "datetime"}
            }
        },
        "log_events": {
            "name": "Application Logs Dataset",
            "schema": {
                "event_id": {"type": "string", "fake_type": "uuid4"},
                "service_name": {"type": "category", "categories": ["web", "api", "db", "cache"]},
                "level": {"type": "category", "categories": ["INFO", "WARN", "ERROR", "DEBUG"]},
                "message": {"type": "string", "max_length": 500},
                "timestamp": {"type": "datetime"}
            }
        }
    },
    "parameters": {
        "compression": {
            "name": "Compression",
            "type": "select",
            "options": ["snappy", "gzip", "none"],
            "default": "snappy"
        },
        "row_group_size": {
            "name": "Row Group Size",
            "type": "number",
            "min": 1000,
            "max": 1000000,
            "default": 100000
        }
    }
}

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_FILENAME = "dataset_generation.log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Create log directory
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
VISUALIZATION = {
    "max_categorical_values": 20,
    "max_correlation_fields": 10,
    "default_chart_height": 400,
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
}

# Task categories
TASK_CATEGORIES = {
    "CSV": ["general", "user", "product"],
    "PARQUET": ["transaction", "iot", "logging"]
}

# Column types
COLUMN_TYPES = {
    "string": ["text", "email", "name", "uuid", "url"],
    "number": ["integer", "float", "percentage"],
    "category": ["single", "multiple"],
    "datetime": ["date", "time", "timestamp"],
    "list": ["text", "number", "mixed"]
}