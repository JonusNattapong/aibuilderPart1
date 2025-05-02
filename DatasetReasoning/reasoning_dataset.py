# -*- coding: utf-8 -*-
"""
Sample dataset for exploring various reasoning approaches in LLMs across different NLP tasks.
Generates separate CSV files for each reasoning type and a combined dataset in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Import data from separate files
from .cot_data import cot_reasoning_data
from .tot_got_data import tot_got_reasoning_data
from .react_data import react_reasoning_data
from .reflection_data import reflection_reasoning_data
from .toolformer_data import toolformer_reasoning_data
from .par_data import par_reasoning_data
from .meta_reasoning_data import meta_reasoning_data

# --- Saving Logic ---
output_dir = 'DataOutput'
os.makedirs(output_dir, exist_ok=True)

# Dictionary mapping data types to their data
data_types = {
    'cot': cot_reasoning_data,
    'tot_got': tot_got_reasoning_data,
    'react': react_reasoning_data,
    'reflection': reflection_reasoning_data,
    'toolformer': toolformer_reasoning_data,
    'par': par_reasoning_data,
    'meta': meta_reasoning_data
}

try:
    # Save individual CSV files for each type
    for data_type, data in data_types.items():
        df = pd.DataFrame(data)
        csv_file = os.path.join(output_dir, f'reasoning_{data_type}_dataset.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Successfully saved {data_type} data to CSV: {csv_file}")
        print(f"Records in {data_type}: {len(data)}")

    # Combine all data for the complete dataset
    reasoning_data = sum(data_types.values(), [])
    
    # Save combined dataset
    combined_df = pd.DataFrame(reasoning_data)
    
    # Save combined CSV
    combined_csv = os.path.join(output_dir, 'reasoning_dataset.csv')
    combined_df.to_csv(combined_csv, index=False, encoding='utf-8')
    print(f"\nSuccessfully saved combined data to CSV: {combined_csv}")
    
    # Save combined Parquet
    combined_parquet = os.path.join(output_dir, 'reasoning_dataset.parquet')
    table = pa.Table.from_pandas(combined_df)
    pq.write_table(table, combined_parquet)
    print(f"Successfully saved combined data to Parquet: {combined_parquet}")

except Exception as e:
    print(f"Error processing or saving data: {e}")

if __name__ == '__main__':
    print(f"\n--- Reasoning Dataset Generation Complete ---")
    print(f"Total records generated: {len(reasoning_data)}")
    # Print summary of records per type
    print("\nRecords per reasoning type:")
    for data_type, data in data_types.items():
        print(f"{data_type}: {len(data)} records")
