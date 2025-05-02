import os
import pandas as pd
import uuid
from config_tabular import (
    OUTPUT_DIR, DEVICE,
    TAB_TO_TEXT_MODEL_ID as MODEL_ID,
    TAB_TO_TEXT_FILENAME as FILENAME,
    TAB_TO_TEXT_N_ROWS as N_ROWS,
    TAB_TO_TEXT_N_COLS as N_COLS,
    TAB_TO_TEXT_NUM_TABLES as NUM_TABLES
)
from tabular_utils import simulate_simple_table, load_t5_model_and_tokenizer, generate_text_from_table

def generate_tab_to_text_dataset(num_tables, n_rows, n_cols, output_dir):
    """Generates Tabular-to-Text data using simulated tables and a local T5 model."""
    print(f"\nGenerating {num_tables} Tabular-to-Text samples locally ({MODEL_ID} on {DEVICE})...")
    data = []

    # Load T5 model and tokenizer
    model, tokenizer = load_t5_model_and_tokenizer(MODEL_ID)
    if not model or not tokenizer:
        print("Failed to load T5 model or tokenizer. Aborting Tabular-to-Text generation.")
        return

    for i in range(num_tables):
        print(f"Processing table {i + 1}/{num_tables}...")

        # Simulate a table
        table_df = simulate_simple_table(n_rows, n_cols)
        table_str = table_df.to_csv(index=False) # Store table as CSV string

        # Generate text description
        generated_text = generate_text_from_table(model, tokenizer, table_df)

        if generated_text is not None:
            data.append({
                'id': str(uuid.uuid4()),
                'table_csv_string': table_str, # Store the table itself as a string
                'generated_text': generated_text
            })
            print(f"  Generated description for table {i+1}")
        else:
            print(f"Warning: Failed to generate text for table {i+1}. Skipping.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        # Ensure CSV string is quoted properly if it contains commas/newlines
        df.to_csv(output_path, index=False, encoding='utf-8', quoting=1) # quoting=1 means QUOTE_MINIMAL
        print(f"Successfully generated and saved {len(data)} Tabular-to-Text samples to {output_path}")
    else:
        print("No Tabular-to-Text data was generated.")

if __name__ == "__main__":
    print("Starting Tabular-to-Text data generation using local T5 model...")
    generate_tab_to_text_dataset(NUM_TABLES, N_ROWS, N_COLS, OUTPUT_DIR)
    print("\nTabular-to-Text data generation process finished.")
