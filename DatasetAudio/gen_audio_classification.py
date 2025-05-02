import os
import pandas as pd
import uuid
import json
from config_audio import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    AUDIO_CLASSIFICATION_MODEL_ID as MODEL_ID,
    AUDIO_CLASSIFICATION_FILENAME as FILENAME,
    AUDIO_CLASSIFICATION_INPUT_AUDIO as INPUT_AUDIO,
    BASE_PATH
)
from audio_utils import invoke_inference_api, load_audio_bytes

def generate_audio_classification_data(num_samples, output_dir):
    """Generates Audio Classification data using HF Inference API."""
    print(f"\nGenerating {num_samples} Audio Classification samples via API ({MODEL_ID})...")
    data = []
    if not INPUT_AUDIO:
        print("Warning: No input audio files configured in config_audio.py. Cannot generate data.")
        return

    num_to_generate = min(num_samples, len(INPUT_AUDIO))

    for i in range(num_to_generate):
        input_audio_path_rel = INPUT_AUDIO[i] # Path relative to project root or placeholder dir
        input_audio_path_abs = os.path.join(BASE_PATH, input_audio_path_rel) # Ensure absolute path for loading
        print(f"Processing sample {i + 1}/{num_to_generate} (Input Audio: {input_audio_path_rel})...")

        audio_bytes = load_audio_bytes(input_audio_path_abs)
        if not audio_bytes:
            print(f"Skipping sample {i+1} due to audio loading error.")
            continue # Skip if audio can't be loaded

        # Call the API
        predictions = invoke_inference_api(MODEL_ID, data=audio_bytes, task="audio-classification")

        if predictions and isinstance(predictions, list):
            try:
                # Store the full list of predictions as a JSON string
                predictions_json = json.dumps(predictions)
                # Optionally, extract the top prediction
                top_prediction = predictions[0] if predictions else {}
                top_label = top_prediction.get('label', 'N/A')
                top_score = top_prediction.get('score', 0.0)

                data.append({
                    'id': str(uuid.uuid4()),
                    'input_audio_path': input_audio_path_rel.replace('\\', '/'), # Use relative path in CSV
                    'top_predicted_label': top_label,
                    'top_predicted_score': top_score,
                    'all_predictions': predictions_json
                })
            except Exception as e:
                 print(f"Warning: Could not process or serialize API result for sample {i+1}: {e}")
                 print(f"API Result: {predictions}")
        else:
            print(f"Warning: API call failed or returned invalid data for sample {i+1}. Skipping. Result: {predictions}")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} Audio Classification samples to {output_path}")
    else:
        print("No Audio Classification data was generated.")

if __name__ == "__main__":
    print("Starting Audio Classification data generation using API...")
    generate_audio_classification_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nAudio Classification data generation process finished.")
