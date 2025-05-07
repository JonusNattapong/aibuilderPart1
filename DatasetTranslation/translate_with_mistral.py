import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from medical_terms import replace_all_terms
from pythainlp import word_tokenize
from pythainlp.util import normalize

# Define paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'ModelUse', 'Mistral-7B-Instruct-v0.3')

def load_model():
    """Load Mistral model and tokenizer."""
    print("Loading model from:", MODEL_PATH)
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize model from local files
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def create_prompt(text: str) -> str:
    """Create a translation prompt."""
    return f"""<s>[INST] You are a medical translator specializing in English to Thai translation.
Translate the following English medical text to Thai while maintaining medical terminology accuracy:

{text}

Thai translation:[/INST]"""

def process_thai_text(text: str) -> str:
    """Process Thai text using PyThaiNLP."""
    # Normalize Thai text
    text = normalize(text)
    
    # Tokenize and join with spaces for better readability
    tokens = word_tokenize(text, engine="newmm")
    return " ".join(tokens)

def translate_with_mistral(text: str, model, tokenizer, max_length=500) -> str:
    """Translate text using Mistral."""
    if not text or not text.strip():
        return ""
        
    try:
        # Create prompt
        prompt = create_prompt(text)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False
        ).to(model.device)
        
        # Generate translation
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract Thai text
        thai_text = translation.split("[/INST]")[-1].strip()
        
        # Process Thai text using PyThaiNLP
        thai_text = process_thai_text(thai_text)
        
        # Apply medical term replacements
        thai_text = replace_all_terms(thai_text)
        
        return thai_text
        
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
        translation = translate_with_mistral(text, model, tokenizer)
        print("\nThai:")
        if translation:
            print(translation)
        else:
            print("Translation failed")
        print("-" * 50)

if __name__ == "__main__":
    main()