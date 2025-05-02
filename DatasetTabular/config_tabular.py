import os

# --- General Configuration ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_PATH, 'DataOutput')
NUM_SAMPLES_PER_TASK = 50 # Number of samples/rows to generate for simulated tasks
DEVICE = "cuda" # Use "cuda" if GPU is available and configured, otherwise "cpu"

# --- Task: Tabular Classification (Simulated) ---
TAB_CLASSIFICATION_FILENAME = "generated_tabular_classification_simulated.csv"
TAB_CLASSIFICATION_N_FEATURES = 5
TAB_CLASSIFICATION_N_CLASSES = 3

# --- Task: Tabular Regression (Simulated) ---
TAB_REGRESSION_FILENAME = "generated_tabular_regression_simulated.csv"
TAB_REGRESSION_N_FEATURES = 4

# --- Task: Tabular-to-Text (Simulated Table + Local T5 Model) ---
TAB_TO_TEXT_MODEL_ID = "t5-small" # Or "t5-base", etc. Needs local download.
TAB_TO_TEXT_FILENAME = "generated_tabular_to_text_local.csv"
TAB_TO_TEXT_N_ROWS = 5 # Number of rows per simulated table
TAB_TO_TEXT_N_COLS = 4 # Number of columns per simulated table
TAB_TO_TEXT_NUM_TABLES = NUM_SAMPLES_PER_TASK // TAB_TO_TEXT_N_ROWS # Generate multiple tables

# --- Task: Time Series Forecasting (Simulated) ---
TIME_SERIES_FILENAME = "generated_time_series_simulated.csv"
TIME_SERIES_LENGTH = 100 # Length of each time series
TIME_SERIES_N_SERIES = NUM_SAMPLES_PER_TASK # Number of different series to generate

# --- Helper to ensure directories exist ---
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

ensure_dirs()
print(f"Tabular Configuration Loaded. Output Dir: {OUTPUT_DIR}")
print(f"Using Device for T5: {DEVICE}")

# Check for necessary libraries (optional)
try:
    import transformers
    import torch
    import pandas
    import sklearn
    import numpy
    print("transformers, torch, pandas, scikit-learn, numpy found.")
except ImportError as e:
    print(f"Warning: Missing core library: {e}. Install with 'pip install transformers torch pandas scikit-learn numpy'")
