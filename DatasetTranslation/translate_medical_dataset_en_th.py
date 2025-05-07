import os
import sys
import torch
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import time

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(BASE_PATH, 'DatasetDownload/FreedomIntelligence_medical-o1-reasoning-SFT')
OUTPUT_PATH = os.path.join(BASE_PATH, 'DatasetDownload/FreedomIntelligence_medical-o1-reasoning-SFT_th')
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "eng_Latn"  # English in Latin script
TARGET_LANG = "tha_Thai"  # Thai language

def load_translator():
    """Loads the translation pipeline."""
    print(f"Loading translation model: {MODEL_NAME}...")
    try:
        # Use local model if available, otherwise download from hub
        model_path = os.path.join(BASE_PATH, "ModelUse", "nllb-200-distilled-600M")
        
        if os.path.exists(model_path):
            print(f"Loading local model from {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            print("Downloading model from Hugging Face Hub")
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Save model locally for future use
            print(f"Saving model to {model_path}")
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Configure the pipeline for NLLB model
        translator = pipeline("translation",
                           model=model,
                           tokenizer=tokenizer,
                           src_lang=SOURCE_LANG,
                           tgt_lang=TARGET_LANG,
                           device=0 if device == "cuda" else -1)
        print(f"Model loaded successfully on {device.upper()}")
        return translator
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        return None

def translate_text(text: str, translator) -> str:
    """Translate a single text from English to Thai."""
    try:
        # Split into smaller chunks (300 chars) to avoid memory issues
        chunks = [text[i:i+300] for i in range(0, len(text), 300)]
        translated_chunks = []
        
        for chunk in chunks:
            try:
                result = translator(chunk,
                                 max_length=400,
                                 src_lang=SOURCE_LANG,
                                 tgt_lang=TARGET_LANG)[0]['translation_text']
                translated_chunks.append(result)
                
                # Clean up CUDA memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as chunk_e:
                print(f"Error translating chunk: {chunk_e}")
                translated_chunks.append(chunk)  # Keep original text for failed chunks
                continue
            
        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"Error in translate_text: {e}")
        return None

def translate_dataset():
    """Translate the medical dataset from English to Thai."""
    print("Loading dataset...")
    try:
        raw_dataset = load_from_disk(DATASET_PATH)
        dataset = raw_dataset['train']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    translator = load_translator()
    if not translator:
        print("Failed to load translator. Aborting.")
        return

    print("Starting translation...")
    translated_data = []
    error_count = 0
    start_time = time.time()
    
    # Check the first item to understand the structure
    first_item = dataset[0]
    print("\nFeatures:", dataset.features)
    print("\nFirst item:", dataset[0])
    
    # Process in smaller batches
    batch_size = 100
    total_items = len(dataset)
    
    for start_idx in tqdm(range(0, total_items, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, total_items)
        
        for idx in range(start_idx, end_idx):
            try:
                item = dataset[idx]
                translated_item = {
                    "id": idx,
                    "question": item["Question"],
                    "question_en": item["Question"],
                    "response": item["Response"],
                    "response_en": item["Response"],
                    "complex_cot": item["Complex_CoT"],
                    "complex_cot_en": item["Complex_CoT"]
                }
                
                # Translate fields
                question_th = translate_text(item["Question"], translator)
                response_th = translate_text(item["Response"], translator)
                cot_th = translate_text(item["Complex_CoT"], translator)
                
                if question_th is None or response_th is None or cot_th is None:
                    error_count += 1
                    print(f"\nError translating item {idx + 1}")
                    if error_count > 10:
                        print("\nToo many translation errors. Aborting.")
                        break
                    continue
                
                translated_item["question"] = question_th
                translated_item["response"] = response_th
                translated_item["complex_cot"] = cot_th
                
                translated_data.append(translated_item)
                
                # Save progress more frequently (every 10 items)
                if (idx + 1) % 10 == 0:
                    print(f"\nProcessed {idx + 1} items. Translation errors: {error_count}")
                    try:
                        save_progress(translated_data)
                        print("Progress saved successfully")
                    except Exception as e:
                        print(f"Error saving progress: {e}")
                    
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nError processing item {idx}: {e}")
                continue

    total_time = time.time() - start_time
    print(f"\nTranslation completed in {total_time:.2f} seconds")
    print(f"Total items processed: {len(dataset)}")
    print(f"Successfully translated: {len(translated_data)}")
    print(f"Translation errors: {error_count}")
    save_progress(translated_data)

def save_progress(data):
    """Save the translated dataset."""
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(OUTPUT_PATH)
    print(f"Dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    print("=== Medical Dataset Translation (EN -> TH) ===")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        print("Please download the English dataset first using:")
        print("python Script/download_hf_datasets.py -d FreedomIntelligence/medical-o1-reasoning-SFT --subset en")
        sys.exit(1)
        
    translate_dataset()
    print("=== Translation Complete ===")