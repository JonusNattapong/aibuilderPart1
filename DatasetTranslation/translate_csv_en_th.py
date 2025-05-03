import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time

# Define paths relative to the script location or a base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
INPUT_CSV = os.path.join(BASE_PATH, 'DataOutput', 'thai_dataset_translation_en_th.csv')
OUTPUT_CSV = os.path.join(BASE_PATH, 'DataOutput', 'translated_en_to_th_from_csv.csv')
MODEL_NAME = "Helsinki-NLP/opus-mt-en-th"

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

def translate_dataframe(df, translator, text_column='english_text'):
    """Translates a specific column in the DataFrame."""
    if translator is None:
        print("Translator not loaded. Skipping translation.")
        return None
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in the DataFrame.")
        return None

    translations = []
    total_rows = len(df)
    print(f"Starting translation for {total_rows} rows...")
    start_time = time.time()

    for index, row in df.iterrows():
        text_to_translate = row[text_column]
        if pd.isna(text_to_translate) or not isinstance(text_to_translate, str) or not text_to_translate.strip():
            translations.append(None) # Handle empty or non-string data
            print(f"  Skipped row {index + 1}/{total_rows} (empty or invalid text)")
            continue
        try:
            translated_text = translator(text_to_translate, max_length=512)[0]['translation_text']
            translations.append(translated_text)
            if (index + 1) % 10 == 0 or (index + 1) == total_rows:
                 elapsed_time = time.time() - start_time
                 print(f"  Translated {index + 1}/{total_rows} rows... (Elapsed: {elapsed_time:.2f}s)")
        except Exception as e:
            print(f"Error translating row {index + 1} ('{text_to_translate[:50]}...'): {e}")
            translations.append(f"Error: {e}")

    print(f"Finished translation. Total time: {time.time() - start_time:.2f}s")
    return translations

if __name__ == "__main__":
    print("--- English to Thai CSV Translation Script ---")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Load data
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV file not found at {INPUT_CSV}")
        print("Please generate it first (e.g., by running Dataset/CSV/translation_en_th_dataset.py)")
    else:
        print(f"Reading data from {INPUT_CSV}...")
        try:
            df = pd.read_csv(INPUT_CSV)

            # Load translator
            translator = load_translator()

            if translator:
                # Translate
                df['translated_thai'] = translate_dataframe(df, translator, text_column='english_text')

                # Save results
                print(f"Saving translated data to {OUTPUT_CSV}...")
                df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
                print("Translation complete and saved.")
            else:
                print("Translation process aborted due to model loading failure.")

        except Exception as e:
            print(f"An error occurred during the process: {e}")

    print("--- Script Finished ---")
