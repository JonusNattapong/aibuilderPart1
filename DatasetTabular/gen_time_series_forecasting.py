import os
import pandas as pd
from config_tabular import (
    OUTPUT_DIR,
    TIME_SERIES_FILENAME as FILENAME,
    TIME_SERIES_LENGTH as SERIES_LENGTH,
    TIME_SERIES_N_SERIES as N_SERIES
)
from tabular_utils import simulate_time_series_data

def generate_time_series_dataset(n_series, series_length, output_dir):
    """Generates simulated Time Series data."""
    print(f"\nGenerating {n_series} simulated Time Series (length {series_length})...")

    df = simulate_time_series_data(n_series, series_length)

    # Save to CSV
    if not df.empty:
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(df)} time series records ({n_series} series) to {output_path}")
    else:
        print("No time series data was generated.")

if __name__ == "__main__":
    print("Starting simulated Time Series data generation...")
    generate_time_series_dataset(N_SERIES, SERIES_LENGTH, OUTPUT_DIR)
    print("\nTime Series data generation process finished.")
