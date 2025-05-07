import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from medical_terms import replace_all_terms
from pythainlp import word_tokenize
from pythainlp.util import normalize

MODEL_ID = "microsoft/phi-2"

def load_model():
    """Load Phi-2 model and tokenizer."""
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def create_prompt(text: str) -> str:
    """Create a translation prompt."""
    return f"""Instruct: You are a medical translator. Translate the following English medical text to Thai. 
Maintain medical terminology accuracy and natural Thai language flow.

English text:
{text}

Thai translation:"""

def process_thai_text(text: str) -> str:
    """Process Thai text using PyThaiNLP."""
    # Normalize Thai text
    text = normalize(text)
    
    # Tokenize and join with spaces for better readability
    tokens = word_tokenize(text, engine="newmm")
    return " ".join(tokens)

def translate_with_phi(text: str, model, tokenizer, max_length=150) -> str:
    """Translate text using Phi-2."""
    if not text or not text.strip():
        return ""
        
    try:
        # Create prompt
        prompt = create_prompt(text)
        
        # Tokenize with truncation
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            return_token_type_ids=False
        ).to(model.device)
        
        # Generate translation with minimal parameters
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_k=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract Thai text
        thai_text = translation.split("Thai translation:")[-1].strip()
        
        # Process Thai text using PyThaiNLP
        thai_text = process_thai_text(thai_text)
        
        # Apply medical term replacements
        thai_text = replace_all_terms(thai_text)
        
        return thai_text
        
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def main():
    print("Initializing medical translation system...")
    
    # Load model
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
        translation = translate_with_phi(text, model, tokenizer)
        print("\nThai:")
        if translation:
            print(translation)
        else:
            print("Translation failed")
        print("-" * 50)

if __name__ == "__main__":
    main()        
    try:
        # Create prompt
        prompt = create_prompt(text)
        
        # Tokenize with truncation
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            return_token_type_ids=False
        ).to(model.device)
        
        # Generate translation with minimal parameters
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_k=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract Thai text
        thai_text = translation.split("Thai translation:")[-1].strip()
        
        # Process Thai text using PyThaiNLP
        thai_text = process_thai_text(thai_text)
        
        # Apply medical term replacements
        thai_text = replace_all_terms(thai_text)
        
        return thai_text
        
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def main():
    print("Initializing medical translation system...")
    
    # Load model
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
        translation = translate_with_phi(text, model, tokenizer)
        print("\nThai:")
        if translation:
            print(translation)
        else:
            print("Translation failed")
        print("-" * 50)

if __name__ == "__main__":
    main()