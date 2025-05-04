"""
Utility functions for dataset translation.
"""
import logging
import os
import json
import csv
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
from deepl import Translator
import langdetect

def setup_logger(log_file: str = None) -> logging.Logger:
    """Setup and configure logger."""
    logger = logging.getLogger('dataset_translation')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return [
        "EN", "DE", "FR", "ES", "IT", "JA", "KO", "ZH", "RU", "PT", 
        "NL", "PL", "TR", "TH", "VI", "ID"
    ]

def detect_language(text: str) -> str:
    """Detect language of text using langdetect."""
    try:
        return langdetect.detect(text).upper()
    except:
        return "UNKNOWN"

def load_dataset(file_obj) -> List[Dict]:
    """Load dataset from various file formats."""
    filename = file_obj.name.lower()
    
    if filename.endswith('.csv'):
        df = pd.read_csv(file_obj)
        return df.to_dict('records')
    elif filename.endswith('.json'):
        return json.load(file_obj)
    elif filename.endswith('.jsonl'):
        return [json.loads(line) for line in file_obj.readlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {filename}")

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using DeepL API."""
    translator = Translator(os.getenv("DEEPL_API_KEY"))
    result = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)
    return result.text

def validate_translation(original: str, translated: str, src_lang: str, tgt_lang: str) -> Tuple[bool, List[str]]:
    """Validate translation quality."""
    issues = []
    
    # Basic length check
    orig_len = len(original)
    trans_len = len(translated)
    if trans_len < orig_len * 0.5 or trans_len > orig_len * 2:
        issues.append("Translation length significantly different from original")
    
    # Language detection check
    detected_lang = detect_language(translated)
    if detected_lang != tgt_lang:
        issues.append(f"Detected language {detected_lang} doesn't match target {tgt_lang}")
    
    # Add more validation checks as needed
    
    return len(issues) == 0, issues

def save_translations(data: List[Dict], filename: str, format: str, output_dir: str) -> str:
    """Save translations in specified format."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_filename}.{format.lower()}")
    
    if format.upper() == "CSV":
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif format.upper() == "JSON":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif format.upper() == "JSONL":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    return output_path