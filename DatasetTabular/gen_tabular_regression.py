import os
import pandas as pd
from config_tabular import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    TAB_REGRESSION_FILENAME as FILENAME,
    TAB_REGRESSION_N_FEATURES as N_FEATURES
)
from tabular_utils import simulate_regression_data

def generate_regression_dataset(num_samples, n_features, output_dir):
    """Generates simulated Tabular Regression data."""
    print(f"\nGenerating {num_samples} simulated Tabular Regression samples...")

    df = simulate_regression_data(num_samples, n_features)

    # Save to CSV
    if not df.empty:
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(df)} regression samples to {output_path}")
    else:
        print("No regression data was generated.")

if __name__ == "__main__":
    print("Starting simulated Tabular Regression data generation...")
    generate_regression_dataset(NUM_SAMPLES_PER_TASK, N_FEATURES, OUTPUT_DIR)
    print("\nTabular Regression data generation process finished.")
