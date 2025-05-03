import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
import argparse
import sys
import json
import itertools
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv() # Added

# Define paths relative to the script location or a base path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR) # Project root
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, 'input') # Default input specific to translation scripts
DEFAULT_OUTPUT_DIR = os.path.join(BASE_PATH, 'output') # Output remains at project level
DEFAULT_MODEL_DIR = os.path.join(BASE_PATH, 'download') # Model remains at project level
MODEL_NAME = "Helsinki-NLP/opus-mt-en-th" # Model identifier for Hugging Face Hub
DEFAULT_BATCH_SIZE = 32   # Translate 32 sentences per batch on GPU/CPU

# Check for webdataset library
try:
    import webdataset as wds
    WDS_AVAILABLE = True
except ImportError:
    WDS_AVAILABLE = False
    print("Error: 'webdataset' library not found. This script requires it.")
    print("Install it with: pip install webdataset")
    # sys.exit(1) # Exit if library not found

def load_translator(batch_size):
    """Loads the translation pipeline."""
    print(f"Loading translation model: {MODEL_NAME}...")
    try:
        # Check if GPU is available
        device_num = 0 if torch.cuda.is_available() else -1
        device_name = torch.cuda.get_device_name(device_num) if device_num == 0 else "CPU"
        print(f"Attempting to load model on device: {'GPU' if device_num == 0 else 'CPU'} ({device_name})...")
        # Pass batch_size to the pipeline
        translator = pipeline("translation", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device_num, batch_size=batch_size)
        print(f"Model loaded successfully on device: {'GPU' if device_num == 0 else 'CPU'}")
        return translator
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have transformers and torch installed (`pip install transformers torch`)")
        if "CUDA out of memory" in str(e):
             print("CUDA out of memory. Try reducing --batch_size.")
        return None

def translate_webdataset(input_shards, output_path, translator, text_key='en.txt', output_key='th.txt', metadata_key='json', batch_size=DEFAULT_BATCH_SIZE):
    """Translates text from input WebDataset shards using batching and writes to an output file (JSON Lines)."""
    if not WDS_AVAILABLE:
        print("WebDataset library not available. Aborting.")
        return

    if translator is None:
        print("Translator not loaded. Skipping translation.")
        return

    print(f"Processing WebDataset shards: {input_shards}")
    print(f"Reading English text from key: '{text_key}'")
    print(f"Writing translated Thai text to key: '{output_key}'")
    print(f"Copying metadata from key: '{metadata_key}' (if exists)")
    print(f"Using translation batch size: {batch_size}")
    print(f"Saving output as JSON Lines to: {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the dataset pipeline
    # Decode text, handle JSON metadata, skip items with errors
    dataset = wds.WebDataset(input_shards).decode("torch").to_tuple(f"__key__", f"{text_key}", f"{metadata_key}", handler=wds.warn_and_continue)

    total_processed = 0
    total_translated = 0
    total_errors = 0
    start_time = time.time()

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Process dataset in batches
            batch_iterator = iter(dataset)
            while True:
                batch_data = list(itertools.islice(batch_iterator, batch_size))
                if not batch_data:
                    break # End of dataset

                texts_to_translate = []
                original_data_map = {} # Map index in texts_to_translate back to original item in batch_data

                # Prepare batch for translation
                for i, item in enumerate(batch_data):
                    key, text_content, metadata = item
                    total_processed += 1
                    if text_content and isinstance(text_content, str) and text_content.strip():
                        original_data_map[len(texts_to_translate)] = item
                        texts_to_translate.append(text_content[:1024]) # Limit input length
                    else:
                        # Handle items with missing/invalid text - log as error/skip
                        total_errors += 1
                        # print(f"Skipping record {key}: Missing or empty text in key '{text_key}'.")

                if not texts_to_translate:
                    # print("No valid text found in this batch.")
                    continue # Move to the next batch

                # Perform batch translation
                try:
                    translated_results = translator(texts_to_translate, max_length=512)
                    # translated_results is a list of dicts like [{'translation_text': '...'}]
                    translated_texts = [res['translation_text'] for res in translated_results]

                    # Write translated batch results
                    for i, translated_text in enumerate(translated_texts):
                        original_item = original_data_map[i]
                        key, original_text, metadata = original_item

                        output_record = {"__key__": key}

                        # Copy metadata
                        if metadata is not None:
                            if isinstance(metadata, dict):
                                output_record.update(metadata)
                            else:
                                try:
                                    output_record.update(json.loads(metadata))
                                except (json.JSONDecodeError, TypeError):
                                    output_record[metadata_key] = metadata

                        output_record[text_key] = original_text # Include original text
                        output_record[output_key] = translated_text
                        outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                        total_translated += 1

                    if total_processed % (batch_size * 10) == 0: # Log progress less frequently
                        elapsed = time.time() - start_time
                        print(f"  Processed {total_processed} records... (Translated: {total_translated}, Errors/Skipped: {total_errors}, Elapsed: {elapsed:.2f}s)")

                except Exception as e:
                    print(f"Error during batch translation: {e}")
                    # Mark all items in this text batch as errors
                    total_errors += len(texts_to_translate)
                    # Optionally write error records to the output file
                    # for i in range(len(texts_to_translate)):
                    #     original_item = original_data_map[i]
                    #     key, original_text, metadata = original_item
                    #     output_record = {"__key__": key, text_key: original_text, output_key: f"Error: {e}"}
                    #     # ... handle metadata ...
                    #     outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')


    except Exception as e:
        print(f"\nAn error occurred during WebDataset processing: {e}")
    finally:
        elapsed = time.time() - start_time
        print(f"\nFinished processing WebDataset.")
        print(f"Total records processed: {total_processed}")
        print(f"Successfully translated and wrote: {total_translated} records.")
        print(f"Encountered errors or skipped: {total_errors} records.")
        print(f"Total time: {elapsed:.2f}s")

if __name__ == "__main__":
    if not WDS_AVAILABLE:
        sys.exit(1) # Exit if library is not installed

    parser = argparse.ArgumentParser(description="Translate English text in WebDataset shards to Thai using batching.")
    parser.add_argument("input_shards", nargs='+', help=f"Path(s) or URL(s) to the input WebDataset shard(s) (e.g., '{os.path.join(os.path.relpath(DEFAULT_INPUT_DIR, BASE_PATH), 'input-{000..001}.tar')}', 'gs://bucket/data-{{0..9}}.tar'). Assumes relative paths are inside '{os.path.relpath(DEFAULT_INPUT_DIR, BASE_PATH)}' if not found.")
    parser.add_argument("-o", "--output_path", default=None,
                        help=f"Path to the output JSON Lines file. Defaults to 'translated_webdataset_output.jsonl' in '{os.path.relpath(DEFAULT_OUTPUT_DIR, BASE_PATH)}'.")
    parser.add_argument("--text_key", default="en.txt",
                        help="Key (extension) in the WebDataset containing English text (default: en.txt).")
    parser.add_argument("--output_key", default="th.txt",
                        help="Key (field name) for the translated Thai text in the output JSON (default: th.txt).")
    parser.add_argument("--metadata_key", default="json",
                        help="Key (extension) for JSON metadata to copy to the output (default: json).")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of sentences to translate in parallel per batch (default: {DEFAULT_BATCH_SIZE}). Adjust based on GPU memory.")


    args = parser.parse_args()

    print("--- English to Thai WebDataset Translation Script (Batched) ---")

    # Ensure input directory exists (optional, good practice)
    os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)

    # Resolve input shard paths (check relative to DEFAULT_INPUT_DIR if needed)
    resolved_shards = []
    for shard_path in args.input_shards:
        # Basic check: if not a URL and not absolute/existing relative to cwd, check in DEFAULT_INPUT_DIR
        if not ('://' in shard_path or os.path.isabs(shard_path) or os.path.exists(shard_path)):
            path_in_default = os.path.join(DEFAULT_INPUT_DIR, shard_path)
            # Note: WebDataset handles brace expansion, so checking existence directly might not work perfectly for patterns.
            # We'll assume if it's relative and doesn't exist, it might be in the default dir.
            # A more robust check would involve trying to list files matching the pattern.
            # For simplicity, we prepend the default dir path if it looks like a relative file path.
            if '{' not in shard_path and '}' not in shard_path: # Simple check if it's not a pattern
                 if os.path.exists(path_in_default):
                     resolved_shards.append(path_in_default)
                     print(f"Input shard '{shard_path}' found in default translation input directory: {path_in_default}")
                 else:
                     resolved_shards.append(shard_path) # Keep original if not found
                     print(f"Warning: Input shard '{shard_path}' not found directly or in '{os.path.relpath(DEFAULT_INPUT_DIR, BASE_PATH)}'.")
            else:
                 # If it's a pattern, assume it might be relative to the default dir if not absolute
                 # Prepend default dir path - WebDataset library will handle resolution
                 resolved_shards.append(os.path.join(DEFAULT_INPUT_DIR, shard_path))


        else:
            resolved_shards.append(shard_path) # Keep absolute paths or URLs as is

    # Determine output path
    output_path = args.output_path
    if not output_path:
        output_filename = "translated_webdataset_output.jsonl" # More specific default name
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, output_filename)
    elif not os.path.isabs(output_path):
         # If output path is relative, place it in DEFAULT_OUTPUT_DIR
         output_path = os.path.join(DEFAULT_OUTPUT_DIR, output_path)

    # Ensure output directory exists and clear existing file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
         print(f"Warning: Output file {output_path} exists and will be overwritten.")
         try:
             os.remove(output_path)
         except OSError as e:
             print(f"Error removing existing output file: {e}. Please remove it manually.")
             sys.exit(1)

    # Load translator
    translator = load_translator(args.batch_size)

    if translator:
        # Translate using resolved shard paths
        translate_webdataset(resolved_shards, output_path, translator,
                             text_key=args.text_key, output_key=args.output_key,
                             metadata_key=args.metadata_key, batch_size=args.batch_size)
    else:
        print("Translation process aborted due to model loading failure.")

    print("--- Script Finished ---")
