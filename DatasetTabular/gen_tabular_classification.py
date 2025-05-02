import os
import pandas as pd
from config_tabular import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    TAB_CLASSIFICATION_FILENAME as FILENAME,
    TAB_CLASSIFICATION_N_FEATURES as N_FEATURES,
    TAB_CLASSIFICATION_N_CLASSES as N_CLASSES
)
from tabular_utils import simulate_classification_data

def generate_classification_dataset(num_samples, n_features, n_classes, output_dir):
    """Generates simulated Tabular Classification data."""
    print(f"\nGenerating {num_samples} simulated Tabular Classification samples...")

    df = simulate_classification_data(num_samples, n_features, n_classes)

    # Save to CSV
    if not df.empty:
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(df)} classification samples to {output_path}")
    else:
        print("No classification data was generated.")

if __name__ == "__main__":
    print("Starting simulated Tabular Classification data generation...")
    generate_classification_dataset(NUM_SAMPLES_PER_TASK, N_FEATURES, N_CLASSES, OUTPUT_DIR)
    print("\nTabular Classification data generation process finished.")
