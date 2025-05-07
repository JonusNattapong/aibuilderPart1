import os
import re
import torch
from datasets import load_from_disk
from transformers.pipelines import pipeline
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from medical_terms import (
    replace_all_terms,
    escape_regex_chars,
    MEDICAL_TERMS,
    ANATOMICAL_TERMS,
    SYMPTOM_TERMS,
    COMPOUND_ANATOMICAL
)

# Define paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_PATH, 'DatasetDownload', 'FreedomIntelligence_medical-o1-reasoning-SFT')
MODEL_PATH = os.path.join(BASE_PATH, 'ModelUse', 'nllb-200-3.3B')
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "tha_Thai"

def check_paths():
    """Check if required paths exist."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

def preprocess_text(text: str) -> str:
    """Preprocess text before translation."""
    # First check if text is not None
    if text is None:
        return ""
    
    # Replace all medical terms
    text = replace_all_terms(text)
    
    # Clean up text
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'\n+', '\n', text)  # Normalize newlines
    text = re.sub(r'([.,!?])\s*', r'\1 ', text)  # Fix punctuation spacing
    text = text.strip()
    
    return text

def translate_text(text: str, translator) -> str:
    """Translate a single text from English to Thai."""
    try:
        # Preprocess text
        text = preprocess_text(text)
        
        # Create translation prompt
        prompt = f"""Translate this English text to Thai:
English: {text}
Thai:"""

        # Generate translation
        result = translator(prompt,
                          max_new_tokens=300,
                          do_sample=True,
                          temperature=0.7,
                          top_p=0.95)[0]['generated_text']
        
        # Extract just the Thai translation (after "Thai:")
        translation = result.split("Thai:")[-1].strip()
        
        # Clean up result
        translation = re.sub(r'\s+', ' ', translation)  # Remove multiple spaces
        translation = re.sub(r'([.,!?])\s*', r'\1 ', translation)  # Fix punctuation
        translation = re.sub(r'\s*\n\s*', '\n', translation)  # Clean up newlines
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return translation if translation else "ไม่สามารถแปลข้อความได้"
        
    except Exception as e:
        print(f"Error translating text: {e}")
        return "เกิดข้อผิดพลาดในการแปล"

def main():
    # Check paths before starting
    check_paths()
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    translator = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if device == "cuda" else -1)
    
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)['train']
    
    print("\nTranslating 5 examples:")
    for idx in range(5):
        item = dataset[idx]
        print(f"\n=== Example {idx + 1} ===")
        
        # Print original text with highlighted terms
        question = item["Question"]
        response = item["Response"]
        
        print("\nQuestion (EN):")
        # Highlight terms in order of length
        all_terms = {**COMPOUND_ANATOMICAL, **ANATOMICAL_TERMS, **SYMPTOM_TERMS, **MEDICAL_TERMS}
        sorted_terms = sorted(all_terms.items(), key=lambda x: len(x[0]), reverse=True)
        for eng, _ in sorted_terms:
            eng_escaped = escape_regex_chars(eng)
            pattern = re.compile(eng_escaped, re.IGNORECASE)
            if pattern.search(question):
                # Handle special regex characters in replacement
                replacement = f"[{eng}]"
                question = pattern.sub(replacement, question)
        print(question)
        
        print("\nQuestion (TH):")
        th_question = translate_text(item["Question"], translator)
        print(th_question)
        
        print("\nResponse (EN):")
        for eng, _ in sorted_terms:
            eng_escaped = escape_regex_chars(eng)
            pattern = re.compile(eng_escaped, re.IGNORECASE)
            if pattern.search(response):
                replacement = f"[{eng}]"
                response = pattern.sub(replacement, response)
        print(response)
        
        print("\nResponse (TH):")
        th_response = translate_text(item["Response"], translator)
        print(th_response)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()