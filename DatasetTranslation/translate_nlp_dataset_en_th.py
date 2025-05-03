import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
import sys
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv() # Added

# Define paths relative to the script location or a base path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR) # Project root
DEFAULT_INPUT_DIR = os.path.join(BASE_PATH, 'DataInput') # Standard input location (though not directly used here)
DEFAULT_OUTPUT_DIR = os.path.join(BASE_PATH, 'output') # Output remains at project level
DEFAULT_MODEL_DIR = os.path.join(BASE_PATH, 'download') # Model remains at project level
# Add DatasetNLP path to sys.path to import the data - Keep this specific path for this script
sys.path.append(os.path.join(BASE_PATH, 'DatasetNLP'))
OUTPUT_CSV = os.path.join(DEFAULT_OUTPUT_DIR, 'translated_en_to_th_from_nlp_dataset.csv') # Use DEFAULT_OUTPUT_DIR
MODEL_NAME = "Helsinki-NLP/opus-mt-en-th" # Model identifier for Hugging Face Hub

# Attempt to import the data
try:
    from translation_nlp_dataset import translation_data
except ImportError:
    print("Error: Could not import 'translation_data' from DatasetNLP.translation_nlp_dataset.")
    print("Ensure the file exists and the path is correct.")
    translation_data = None # Set to None to prevent further errors

def load_translator():
    """Loads the translation pipeline."""
    print(f"Loading translation model: {MODEL_NAME}...")
    try:
        # Check if GPU is available
        device = 0 if torch.cuda.is_available() else -1
        translator = pipeline("translation", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)
        print(f"Model loaded successfully on device: {'GPU' if device == 0 else 'CPU'}")
        return translator
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have transformers and torch installed (`pip install transformers torch`)")
        return None

def translate_list(data_list, translator):
    """Translates English sentences ('en' key) in a list of dictionaries."""
    if translator is None:
        print("Translator not loaded. Skipping translation.")
        return None
    if not data_list:
        print("Input data list is empty or could not be loaded.")
        return None

    results = []
    total_items = len(data_list)
    print(f"Starting translation for {total_items} items...")
    start_time = time.time()

    for i, item in enumerate(data_list):
        if 'en' in item and isinstance(item['en'], str) and item['en'].strip():
            text_to_translate = item['en']
            try:
                translated_text = translator(text_to_translate, max_length=512)[0]['translation_text']
                results.append({
                    'original_english': text_to_translate,
                    'original_thai': item.get('th', None), # Include original Thai if present
                    'translated_thai': translated_text
                })
                if (i + 1) % 5 == 0 or (i + 1) == total_items:
                     elapsed_time = time.time() - start_time
                     print(f"  Translated {i + 1}/{total_items} items... (Elapsed: {elapsed_time:.2f}s)")
            except Exception as e:
                print(f"Error translating item {i + 1} ('{text_to_translate[:50]}...'): {e}")
                results.append({
                    'original_english': text_to_translate,
                    'original_thai': item.get('th', None),
                    'translated_thai': f"Error: {e}"
                })
        else:
            # Skip items without a valid 'en' key
            print(f"  Skipped item {i + 1}/{total_items} (missing or invalid 'en' text)")
            results.append({
                'original_english': item.get('en', None),
                'original_thai': item.get('th', None),
                'translated_thai': None
            })


    print(f"Finished translation. Total time: {time.time() - start_time:.2f}s")
    return results

if __name__ == "__main__":
    print("--- English to Thai NLP Dataset Translation Script ---")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    if translation_data is None:
        print("Translation data not loaded. Aborting.")
    else:
        # Load translator
        translator = load_translator()

        if translator:
            # Translate
            translated_results = translate_list(translation_data, translator)

            if translated_results:
                # Save results
                print(f"Saving translated data to {OUTPUT_CSV}...")
                df = pd.DataFrame(translated_results)
                df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
                print("Translation complete and saved.")
            else:
                print("No results were translated.")
        else:
            print("Translation process aborted due to model loading failure.")

    print("--- Script Finished ---")
