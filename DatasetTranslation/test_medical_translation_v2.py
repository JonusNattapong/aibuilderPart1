import os
import re
import sys
import gc
import torch
from tqdm import tqdm
from contextlib import nullcontext
from datasets import load_from_disk

# Install required packages
print("Installing dependencies...")
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "pythainlp", "python-crfsuite", "tqdm"])

from pythainlp import word_tokenize
from pythainlp.util import normalize
from transformers.pipelines import pipeline
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from medical_terms import (
    replace_all_terms,
    escape_regex_chars,
    MEDICAL_TERMS,
    ANATOMICAL_TERMS,
    SYMPTOM_TERMS,
    COMPOUND_ANATOMICAL
)

# Define paths using raw strings
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_PATH, 'DatasetDownload', 'FreedomIntelligence_medical-o1-reasoning-SFT')
MODEL_PATH = os.path.join(BASE_PATH, 'ModelUse', 'nllb-200-3.3B')
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "tha_Thai"

def load_model_and_tokenizer():
    """Load the translation model and tokenizer."""
    print("Loading model...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        print("Loading local model:", os.path.basename(MODEL_PATH))
        # Basic memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load NLLB model with optimizations
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("Model loaded successfully")
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        print("Tokenizer loaded successfully")
        
        # Configure tokenizer for translation
        tokenizer.src_lang = SOURCE_LANG
        tokenizer.tgt_lang = TARGET_LANG
        tokenizer._src_lang = SOURCE_LANG
        tokenizer._tgt_lang = TARGET_LANG
        
        # Create translation pipeline
        translator = pipeline(
            task="translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=SOURCE_LANG,
            tgt_lang=TARGET_LANG,
            batch_size=1
        )
        
        print(f"Model loaded successfully on {model.device}")
        return translator, model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def clean_thai_text(text: str) -> str:
    """Clean and normalize Thai text using PyThaiNLP."""
    if not text or not text.strip():
        return ""
    
    try:
        # Remove translation artifacts
        text = re.sub(r'แปล.*?:|Translate.*?:|Thai:', '', text)
        
        # Basic Thai text cleaning
        text = normalize(text)
        tokens = word_tokenize(text)
        text = ' '.join(tokens) if tokens else text
        
        # Clean up spacing
        text = re.sub(r'\s*\n\s*', '\n', text)  # Clean newlines
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = re.sub(r'([.,!?])\s*', r'\1 ', text)  # Fix punctuation spacing
        
        return text.strip()
    except Exception as e:
        print(f"Thai text processing failed: {e}")
        return text.strip()

def translate_with_prompt(text: str, translator, tokenizer=None, model=None) -> str:
    """Translate text using a medical-specific prompt."""
    if not text or not text.strip():
        return ""
        
    try:
        # Simple sentence splitting
        # Split text into chunks for translation
        chunks = []
        current_chunk = ""
        
        # Process each sentence with PyThaiNLP
        # Split into smaller chunks to manage memory better
        sentences = re.split(r'([.!?]+)', text.strip())
        max_chunk_size = 80  # Reduced chunk size
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i].strip()
            if not sent:
                continue
                
            # Add punctuation back if it exists
            if i + 1 < len(sentences):
                sent += sentences[i + 1]
                
            if len(current_chunk) + len(sent) < max_chunk_size:
                current_chunk += sent
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent
                
        if current_chunk:
            chunks.append(current_chunk)
        translated_parts = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            # Translate chunk
            try:
                if tokenizer and model:
                    # Prepare input with explicit language tokens
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True)
                    
                    # Add target language token
                    forced_bos_token = tokenizer.convert_tokens_to_ids(TARGET_LANG)
                    
                    # Generate translation
                    # Get the right token ID for target language
                    if hasattr(tokenizer, 'lang_code_to_id'):
                        tgt_token_id = tokenizer.lang_code_to_id[TARGET_LANG]
                    else:
                        tgt_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

                    # Generate with proper sampling parameters
                    translated = model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_token_id,
                        max_length=100,
                        num_beams=3,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.7,
                        early_stopping=True,
                        # Add memory optimization
                        use_cache=True if torch.cuda.is_available() else False
                    )

                    # Force memory cleanup
                    del inputs
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Decode translation
                    translation = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                else:
                    # Use pipeline
                    translation = translator(
                        chunk,
                        src_lang=SOURCE_LANG,
                        tgt_lang=TARGET_LANG,
                        max_length=150
                    )[0]['translation_text']
                
                # Process Thai text
                result = clean_thai_text(translation) if translation else "เกิดข้อผิดพลาดในการแปล"
                
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Translation error: {e}")
                result = "เกิดข้อผิดพลาดในการแปล"
                continue
            
            # Apply medical term replacements
            result = replace_all_terms(result)
            translated_parts.append(result)
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return ' '.join(translated_parts)
        
    except Exception as e:
        print(f"Translation error: {e}")
        return "เกิดข้อผิดพลาดในการแปล"

def main():
    # Load model and components
    result = load_model_and_tokenizer()
    if not result:
        print("Failed to load translator. Exiting.")
        return
    
    translator, model, tokenizer = result
    
    print("Loading dataset...")
    try:
        dataset = load_from_disk(DATASET_PATH)['train']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print("\nTranslating 5 examples:")
    for idx in tqdm(range(5), desc="Translating examples"):
        try:
            item = dataset[idx]
            print(f"\n=== Example {idx + 1} ===")
            
            # Process question
            question = item["Question"]
            print("\nQuestion (EN):")
            print(question)
            
            # Highlight medical terms
            highlighted_question = question
            all_terms = {**COMPOUND_ANATOMICAL, **ANATOMICAL_TERMS, **SYMPTOM_TERMS, **MEDICAL_TERMS}
            for term in sorted(all_terms.keys(), key=len, reverse=True):
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                if pattern.search(highlighted_question):
                    highlighted_question = pattern.sub(f"[{term}]", highlighted_question)
            print("\nQuestion (EN) with highlighted terms:")
            print(highlighted_question)
            
            # Translate question
            print("\nQuestion (TH):")
            question_th = translate_with_prompt(question, translator, tokenizer, model)
            if question_th:
                print(question_th)
            else:
                print("Translation failed")
            
            # Process response
            response = item["Response"]
            print("\nResponse (EN):")
            print(response)
            
            # Highlight medical terms in response
            highlighted_response = response
            for term in sorted(all_terms.keys(), key=len, reverse=True):
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                if pattern.search(highlighted_response):
                    highlighted_response = pattern.sub(f"[{term}]", highlighted_response)
            print("\nResponse (EN) with highlighted terms:")
            print(highlighted_response)
            
            # Translate response
            print("\nResponse (TH):")
            response_th = translate_with_prompt(response, translator, tokenizer, model)
            if response_th:
                print(response_th)
            else:
                print("Translation failed")
                
            print("\n" + "="*50)
            
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

if __name__ == "__main__":
    main()