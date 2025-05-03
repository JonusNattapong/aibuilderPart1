import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
import argparse
import sys
import math

# Define paths relative to the script location or a base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
DEFAULT_OUTPUT_DIR = os.path.join(BASE_PATH, 'DataOutput')
MODEL_NAME = "Helsinki-NLP/opus-mt-en-th"
DEFAULT_CHUNK_SIZE = 1000 # Process 1000 rows at a time
DEFAULT_BATCH_SIZE = 32   # Translate 32 sentences per batch on GPU/CPU

# Ensure pyarrow is installed for Parquet support
try:
    import pyarrow
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    # print("Warning: 'pyarrow' library not found. Parquet file support will be disabled.")
    # print("Install it with: pip install pyarrow")


def load_translator(batch_size):
    """Loads the translation pipeline."""
    print(f"Loading translation model: {MODEL_NAME}...")
    try:
        # Check if GPU is available
        device_num = 0 if torch.cuda.is_available() else -1
        device_name = torch.cuda.get_device_name(device_num) if device_num == 0 else "CPU"
        print(f"Attempting to load model on device: {'GPU' if device_num == 0 else 'CPU'} ({device_name})...")
        translator = pipeline("translation", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device_num, batch_size=batch_size)
        print(f"Model loaded successfully on device: {'GPU' if device_num == 0 else 'CPU'}")
        return translator
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have transformers and torch installed (`pip install transformers torch`)")
        if "CUDA out of memory" in str(e):
             print("CUDA out of memory. Try reducing --batch_size.")
        return None

def translate_chunk(chunk_df, translator, text_column='english_text', batch_size=DEFAULT_BATCH_SIZE):
    """Translates a specific column in a DataFrame chunk using batching."""
    if translator is None:
        print("Translator not loaded. Skipping translation.")
        return None
    if text_column not in chunk_df.columns:
        print(f"Error: Column '{text_column}' not found in the DataFrame chunk.")
        return None

    # Prepare list of texts to translate, handling potential NaN/invalid values
    texts_to_translate = []
    original_indices = []
    valid_texts_map = {} # Map index in texts_to_translate back to original chunk index

    for idx, text in chunk_df[text_column].items():
        if pd.notna(text) and isinstance(text, str) and text.strip():
            valid_texts_map[len(texts_to_translate)] = idx
            texts_to_translate.append(text[:1024]) # Limit input length per item
        else:
            # Store original index to place None later
            original_indices.append(idx)

    if not texts_to_translate:
        # print("No valid text found in this chunk.")
        return pd.Series([None] * len(chunk_df), index=chunk_df.index) # Return series of Nones matching chunk index

    # print(f"  Translating {len(texts_to_translate)} valid texts in chunk...")
    translations_list = []
    try:
        # Use the pipeline's batching
        results_generator = translator(texts_to_translate, max_length=512)
        for result in results_generator:
             translations_list.append(result['translation_text'])

    except Exception as e:
        print(f"Error during batch translation: {e}")
        # Fallback or return errors - let's return error string for affected items
        # This simplistic approach might misalign if only part of a batch fails.
        # A more robust solution would handle errors per item if the pipeline allows.
        return pd.Series([f"Error: {e}"] * len(chunk_df), index=chunk_df.index)

    # Reconstruct the translation series with original indices, inserting Nones/Errors
    translated_series = pd.Series([None] * len(chunk_df), index=chunk_df.index, dtype=object)
    for i, translated_text in enumerate(translations_list):
        original_idx = valid_texts_map[i]
        translated_series.loc[original_idx] = translated_text

    # print(f"  Finished translating chunk.")
    return translated_series


def process_file_in_chunks(input_path, input_format, output_path, output_format, translator, text_column, output_column, chunk_size, batch_size):
    """Reads, translates, and saves the file chunk by chunk."""
    print(f"Starting chunk processing: Input='{input_path}', Output='{output_path}', ChunkSize={chunk_size}, BatchSize={batch_size}")
    start_time = time.time()
    total_rows_processed = 0
    is_first_chunk = True

    try:
        # Determine reader based on format
        if input_format == 'csv':
            reader = pd.read_csv(input_path, chunksize=chunk_size, iterator=True)
        elif input_format == 'json':
            # JSON chunking works best with lines=True
            try:
                reader = pd.read_json(input_path, lines=True, chunksize=chunk_size, iterator=True)
            except ValueError:
                print("Warning: Could not read JSON as lines. Reading entire file - may consume large memory.")
                # Fallback: Read all, then chunk the DataFrame (Memory intensive!)
                df_full = pd.read_json(input_path)
                num_chunks = math.ceil(len(df_full) / chunk_size)
                reader = (df_full[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks))
        elif input_format == 'parquet':
            if not PYARROW_AVAILABLE:
                print("Error: 'pyarrow' is required for Parquet. Install it with `pip install pyarrow`.")
                return
            print("Warning: True streaming for Parquet is complex with Pandas. Reading file then iterating chunks (Memory intensive for large files). Consider partitioned Parquet or Dask for very large files.")
            # Fallback: Read all, then chunk (Memory intensive!)
            # For true streaming, would need pyarrow.parquet.ParquetFile and read row groups.
            df_full = pd.read_parquet(input_path)
            num_chunks = math.ceil(len(df_full) / chunk_size)
            reader = (df_full[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks))
        else:
            print(f"Error: Unsupported input format '{input_format}' for chunk processing.")
            return

        # Process each chunk
        for i, chunk_df in enumerate(reader):
            chunk_start_time = time.time()
            print(f"Processing chunk {i + 1} ({len(chunk_df)} rows)...")

            # Translate the chunk
            translated_col = translate_chunk(chunk_df, translator, text_column, batch_size)
            if translated_col is None:
                print(f"Skipping chunk {i+1} due to translation error.")
                continue

            chunk_df[output_column] = translated_col
            rows_in_chunk = len(chunk_df)
            total_rows_processed += rows_in_chunk

            # Save the processed chunk
            save_mode = 'w' if is_first_chunk else 'a'
            include_header = is_first_chunk

            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if output_format == 'csv':
                    chunk_df.to_csv(output_path, mode=save_mode, header=include_header, index=False, encoding='utf-8')
                elif output_format == 'json':
                    # Append mode for JSON lines
                    chunk_df.to_json(output_path, orient='records', lines=True, mode=save_mode, force_ascii=False)
                elif output_format == 'parquet':
                    if not PYARROW_AVAILABLE:
                        print("Error: 'pyarrow' required for Parquet.")
                        continue # Skip saving this chunk
                    # Parquet append is tricky, easier to write separate files per chunk or collect and write once if feasible
                    # For simplicity here, we overwrite or append using pyarrow if possible, but this might not be efficient
                    # A better approach for huge parquet is often writing partitioned files.
                    # Let's use pandas append capability which might rewrite data.
                    if is_first_chunk:
                         chunk_df.to_parquet(output_path, index=False)
                    else:
                         # Read existing, append, write back (Inefficient for large files!)
                         # Consider Dask or writing chunked files like output_chunk_0.parquet, output_chunk_1.parquet...
                         # For now, demonstrating the concept with potential inefficiency:
                         try:
                             existing_df = pd.read_parquet(output_path)
                             combined_df = pd.concat([existing_df, chunk_df], ignore_index=True)
                             combined_df.to_parquet(output_path, index=False)
                         except Exception as pq_err:
                             print(f"  Error appending Parquet chunk (might be large file issue): {pq_err}. Consider writing separate chunk files.")

                is_first_chunk = False # Only write header for the first chunk
                chunk_time = time.time() - chunk_start_time
                print(f"  Chunk {i + 1} finished in {chunk_time:.2f}s. Total rows processed: {total_rows_processed}")

            except Exception as e:
                print(f"Error saving chunk {i + 1} to {output_path}: {e}")
                # Decide whether to continue or stop
                # continue

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred during chunk processing: {e}")
    finally:
        total_time = time.time() - start_time
        print(f"\nFinished processing all chunks.")
        print(f"Total rows processed: {total_rows_processed}")
        print(f"Total time: {total_time:.2f}s")


# Removed old read_data, save_data, translate_dataframe functions as logic is now in chunk processing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English text column in large CSV, JSON, or Parquet files to Thai using chunking.")
    parser.add_argument("input_path", help="Path to the input file (CSV, JSON Lines, Parquet).")
    parser.add_argument("-o", "--output_path", default=None,
                        help=f"Path to the output file. Defaults to input filename with '_translated_th' suffix in '{DEFAULT_OUTPUT_DIR}'.")
    parser.add_argument("-if", "--input_format", choices=['csv', 'json', 'parquet'], default=None,
                        help="Format of the input file. If not provided, attempts to infer from file extension.")
    parser.add_argument("-of", "--output_format", choices=['csv', 'json', 'parquet'], default='csv',
                        help="Format for the output file (default: csv). Parquet append might be inefficient.")
    parser.add_argument("-c", "--text_column", default="english_text",
                        help="Name of the column containing English text to translate (default: english_text).")
    parser.add_argument("--output_column", default="translated_thai",
                        help="Name for the new column containing translated Thai text (default: translated_thai).")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Number of rows to process per chunk (default: {DEFAULT_CHUNK_SIZE}).")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of sentences to translate in parallel per batch (default: {DEFAULT_BATCH_SIZE}). Adjust based on GPU memory.")


    args = parser.parse_args()

    print("--- English to Thai Tabular File Translation Script (Chunked) ---")

    # Determine input format
    input_format = args.input_format
    if not input_format:
        _, ext = os.path.splitext(args.input_path)
        ext = ext.lower()
        if ext == '.csv':
            input_format = 'csv'
        elif ext == '.json' or ext == '.jsonl':
            input_format = 'json' # Assume JSON Lines for chunking
        elif ext == '.parquet':
            input_format = 'parquet'
        else:
            print(f"Error: Could not infer input format from file extension '{ext}'. Please specify using --input_format.")
            sys.exit(1)

    # Determine output path
    output_path = args.output_path
    if not output_path:
        input_filename = os.path.basename(args.input_path)
        name, _ = os.path.splitext(input_filename)
        output_filename = f"{name}_translated_th.{args.output_format}"
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, output_filename)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Delete existing output file before starting chunk processing if it exists
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
        # Process the file in chunks
        process_file_in_chunks(
            args.input_path, input_format, output_path, args.output_format,
            translator, args.text_column, args.output_column,
            args.chunk_size, args.batch_size
        )
    else:
        print("Translation process aborted due to model loading failure.")

    print("--- Script Finished ---")
