import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from medical_terms import replace_all_terms

# Define paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'ModelUse', 'nllb-200-distilled-600M')

def load_model():
    """Load NLLB model and tokenizer."""
    print("Loading model from:", MODEL_PATH)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            src_lang="eng_Latn",
            tgt_lang="tha_Thai"
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def translate_with_nllb(text: str, model, tokenizer, max_length=500) -> str:
    """Translate text using NLLB."""
    if not text or not text.strip():
        return ""
        
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        # Generate translation
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["tha_Thai"],
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        # Decode translation
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        # Apply medical term replacements
        translation = replace_all_terms(translation)
        
        return translation
        
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def main():
    # Load model
    print("Initializing medical translation system...")
    model, tokenizer = load_model()
    if not model or not tokenizer:
        return
        
    # Test translations
    test_texts = [
        "The patient presents with acute myocardial infarction and severe chest pain.",
        "MRI reveals multiple sclerosis lesions in the cerebral cortex.",
        "The patient exhibits symptoms of Parkinson's disease including tremors and bradykinesia."
    ]
    
    print("\nTesting medical translations:")
    for text in test_texts:
        print("\nEnglish:")
        print(text)
        
        print("Translating...")
        translation = translate_with_nllb(text, model, tokenizer)
        print("\nThai:")
        if translation:
            print(translation)
        else:
            print("Translation failed")
        print("-" * 50)

if __name__ == "__main__":
    main()