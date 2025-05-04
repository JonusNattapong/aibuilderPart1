"""
Shared utilities for both API and Streamlit app.
"""
import logging
import os
import json
import asyncio
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
from deepl import Translator
import langdetect

logger = logging.getLogger(__name__)

class TranslationManager:
    """Manages translation operations for both API and Streamlit interfaces."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPL_API_KEY")
        if not self.api_key:
            raise ValueError("DeepL API key not provided")
        self.translator = Translator(self.api_key)
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "EN", "DE", "FR", "ES", "IT", "JA", "KO", "ZH", "RU", "PT", 
            "NL", "PL", "TR", "TH", "VI", "ID"
        ]

    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            return langdetect.detect(text).upper()
        except:
            return "UNKNOWN"

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """Translate single text with validation."""
        try:
            # Translate text
            translated = self.translator.translate_text(
                text, 
                source_lang=source_lang, 
                target_lang=target_lang
            ).text

            # Validate translation
            is_valid, issues = await self.validate_translation(
                text, translated, source_lang, target_lang
            )

            return {
                "original": text,
                "translated": translated,
                "detected_lang": self.detect_language(text),
                "needs_review": not is_valid,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "original": text,
                "error": str(e)
            }

    async def validate_translation(self, 
                                 original: str, 
                                 translated: str,
                                 source_lang: str, 
                                 target_lang: str) -> Tuple[bool, List[str]]:
        """Validate translation quality."""
        issues = []
        
        # Basic length check
        orig_len = len(original)
        trans_len = len(translated)
        if trans_len < orig_len * 0.5 or trans_len > orig_len * 2:
            issues.append("Translation length significantly different from original")
        
        # Language detection check
        detected_lang = self.detect_language(translated)
        if detected_lang != target_lang:
            issues.append(f"Detected language {detected_lang} doesn't match target {target_lang}")
        
        return len(issues) == 0, issues

    async def translate_batch(self, 
                            texts: List[str],
                            source_lang: str,
                            target_lang: str,
                            batch_size: int = 10,
                            progress_callback=None) -> List[Dict]:
        """Translate a batch of texts with progress tracking."""
        results = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            tasks = []
            for text in batch:
                task = asyncio.create_task(
                    self.translate_text(text, source_lang, target_lang)
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            if progress_callback:
                progress = (i + len(batch)) / total
                progress_callback(progress)

        return results

    def load_dataset(self, file_obj) -> List[Dict]:
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

    def save_translations(self, 
                        data: List[Dict],
                        filename: str,
                        format: str,
                        output_dir: str) -> str:
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