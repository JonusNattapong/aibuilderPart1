import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config_tabular import DEVICE

# --- Model Loading Cache (for T5) ---
_model_cache = {}
_tokenizer_cache = {}

def load_t5_model_and_tokenizer(model_id):
    """Loads a T5 model and tokenizer, caching them."""
    global _model_cache, _tokenizer_cache
    if model_id in _model_cache and model_id in _tokenizer_cache:
        print(f"Using cached T5 model and tokenizer for {model_id}")
        return _model_cache[model_id], _tokenizer_cache[model_id]

    print(f"Loading T5 model and tokenizer for {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(DEVICE)
        model.eval()
        _tokenizer_cache[model_id] = tokenizer
        _model_cache[model_id] = model
        print(f"Successfully loaded {model_id} to {DEVICE}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading T5 model/tokenizer {model_id}: {e}")
        return None, None

# --- Data Simulation Functions ---

def simulate_classification_data(n_samples, n_features, n_classes):
    """Simulates tabular data for classification using sklearn."""
    print(f"Simulating classification data: {n_samples} samples, {n_features} features, {n_classes} classes")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2), # Ensure at least 2 informative features
        n_redundant=max(0, n_features // 4),
        n_classes=n_classes,
        random_state=42
    )
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target_class'] = y
    return df

def simulate_regression_data(n_samples, n_features):
    """Simulates tabular data for regression using sklearn."""
    print(f"Simulating regression data: {n_samples} samples, {n_features} features")
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2), # Ensure at least 2 informative features
        noise=10.0, # Add some noise
        random_state=42
    )
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target_value'] = y
    return df

def simulate_simple_table(n_rows, n_cols):
    """Simulates a simple pandas DataFrame with random data."""
    data = np.random.rand(n_rows, n_cols)
    col_names = [f'Column_{chr(65+i)}' for i in range(n_cols)] # Column_A, Column_B, ...
    df = pd.DataFrame(data, columns=col_names)
    # Add some categorical data for variety
    if n_cols > 1:
        categories = ['Type1', 'Type2', 'Type3']
        df['Category'] = np.random.choice(categories, size=n_rows)
        # Shuffle columns to mix types
        df = df[np.random.permutation(df.columns)]
    return df

def simulate_time_series_data(n_series, series_length):
    """Simulates multiple simple time series (e.g., sine wave + noise)."""
    print(f"Simulating {n_series} time series, each of length {series_length}")
    all_series_data = {}
    time_index = pd.date_range(start='2023-01-01', periods=series_length, freq='D')

    for i in range(n_series):
        series_id = f'series_{i+1}'
        # Create a base sine wave with varying frequency/amplitude
        frequency = np.random.uniform(0.05, 0.2)
        amplitude = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2 * np.pi)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * np.arange(series_length) + phase)

        # Add some random noise
        noise = np.random.normal(0, 0.2, series_length)

        # Add a slight trend
        trend = np.linspace(0, np.random.uniform(-0.5, 0.5), series_length)

        series_values = sine_wave + noise + trend + np.random.uniform(1, 5) # Add base value

        df = pd.DataFrame({'timestamp': time_index, 'value': series_values, 'series_id': series_id})
        all_series_data[series_id] = df

    # Combine into a single DataFrame
    combined_df = pd.concat(all_series_data.values(), ignore_index=True)
    return combined_df


# --- Tabular-to-Text Generation ---

def generate_text_from_table(model, tokenizer, table_df):
    """Generates a textual description of a pandas DataFrame using a T5 model."""
    try:
        # Simple serialization: convert table to string format
        # More sophisticated serialization might be needed for better results
        table_str = table_df.to_string(index=False, float_format='%.2f')
        prompt = f"Describe the following table:\n{table_str}"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

        # Generate text
        with torch.no_grad():
            # Adjust generation parameters as needed
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=150, # Limit output length
                num_beams=4,    # Use beam search
                early_stopping=True
            )

        # Decode the generated text
        generated_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as e:
        print(f"Error during Tabular-to-Text generation: {e}")
        return None
