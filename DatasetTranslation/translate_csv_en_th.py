import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
import argparse # Add argparse for flexibility
import sys

# Define paths relative to the script location or a base path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR) # Project root
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, 'input') # Default input specific to translation scripts
DEFAULT_OUTPUT_DIR = os.path.join(BASE_PATH, 'output') # Output remains at project level
DEFAULT_MODEL_DIR = os.path.join(BASE_PATH, 'download') # Model remains at project level
# Default input CSV name, expected in DEFAULT_INPUT_DIR
DEFAULT_INPUT_FILENAME = 'thai_dataset_translation_en_th.csv'
# Default output CSV name, will be saved in DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_FILENAME = 'translated_en_to_th_from_csv.csv'
MODEL_NAME = "Helsinki-NLP/opus-mt-en-th" # Model identifier for Hugging Face Hub
DEFAULT_BATCH_SIZE = 32 # Add batch size for consistency

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
    parser = argparse.ArgumentParser(description="Translate an English text column in a CSV file to Thai using batching.")
    parser.add_argument("-i", "--input_path", default=os.path.join(DEFAULT_INPUT_DIR, DEFAULT_INPUT_FILENAME),
                        help=f"Path to the input CSV file. Defaults to '{DEFAULT_INPUT_FILENAME}' in '{os.path.relpath(DEFAULT_INPUT_DIR, BASE_PATH)}'.")
    parser.add_argument("-o", "--output_path", default=os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME),
                        help=f"Path to the output CSV file. Defaults to '{DEFAULT_OUTPUT_FILENAME}' in '{os.path.relpath(DEFAULT_OUTPUT_DIR, BASE_PATH)}'.")

    args = parser.parse_args()

    print("--- English to Thai CSV Translation Script (Batched) ---")

    # Ensure input and output directories exist
    os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Resolve input path (similar logic to tabular script)
    input_path_resolved = args.input_path
    if not os.path.isabs(input_path_resolved) and not os.path.exists(input_path_resolved):
        # Check if the default path (which includes DEFAULT_INPUT_DIR) exists
        default_path_check = os.path.join(DEFAULT_INPUT_DIR, DEFAULT_INPUT_FILENAME)
        if args.input_path == default_path_check and os.path.exists(default_path_check):
             input_path_resolved = default_path_check
             # print(f"Using default input file: {input_path_resolved}") # Already default
        elif os.path.exists(os.path.join(DEFAULT_INPUT_DIR, args.input_path)):
             input_path_resolved = os.path.join(DEFAULT_INPUT_DIR, args.input_path)
             print(f"Input file found in default translation input directory: {input_path_resolved}")
        else:
            # Keep original path, let pd.read_csv handle FileNotFoundError
            print(f"Warning: Input file not found at '{args.input_path}' or in '{os.path.relpath(DEFAULT_INPUT_DIR, BASE_PATH)}'.")


    # Load data using the resolved path
    if not os.path.exists(input_path_resolved):
        print(f"Error: Input CSV file not found at {input_path_resolved}")
        print(f"Please ensure the file exists or generate it (e.g., by running Dataset/CSV/translation_en_th_dataset.py and placing it in {os.path.relpath(DEFAULT_INPUT_DIR, BASE_PATH)}).")
        sys.exit(1)
    else:
        print(f"Reading data from {input_path_resolved}...")
        try:
            df = pd.read_csv(input_path_resolved)

            # Load translator
            translator = load_translator()

            if translator:
                # Translate
                df['translated_thai'] = translate_dataframe(df, translator, text_column='english_text')

                # Save results
                print(f"Saving translated data to {args.output_path}...")
                df.to_csv(args.output_path, index=False, encoding='utf-8')
                print("Translation complete and saved.")
            else:
                print("Translation process aborted due to model loading failure.")

        except Exception as e:
            print(f"An error occurred during the process: {e}")

    print("--- Script Finished ---")
