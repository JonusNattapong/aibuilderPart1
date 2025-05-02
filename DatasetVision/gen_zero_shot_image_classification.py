import os
import pandas as pd
import json
from config_vision import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    ZERO_SHOT_IMAGE_CLASSIFICATION_MODEL_ID as MODEL_ID,
    ZERO_SHOT_IMAGE_CLASSIFICATION_FILENAME as FILENAME,
    ZERO_SHOT_IMAGE_CLASSIFICATION_INPUT_IMAGES as INPUT_IMAGES,
    ZERO_SHOT_IMAGE_CLASSIFICATION_CANDIDATES as CANDIDATE_LABELS_LIST
)
from vision_utils import invoke_inference_api, load_image_bytes

def generate_zero_shot_image_classification_data(num_samples, output_dir):
    """Generates Zero-Shot Image Classification data using HF Inference API."""
    print(f"\nGenerating {num_samples} zero-shot image classification samples via API ({MODEL_ID})...")
    data = []
    if not INPUT_IMAGES or not CANDIDATE_LABELS_LIST:
        print("Warning: No input images or candidate labels configured in config_vision.py. Cannot generate data.")
        return

    num_to_generate = min(num_samples, len(INPUT_IMAGES), len(CANDIDATE_LABELS_LIST))

    for i in range(num_to_generate):
        input_image_path = INPUT_IMAGES[i]
        candidate_labels = CANDIDATE_LABELS_LIST[i]
        print(f"Processing sample {i + 1}/{num_to_generate} (Input: {input_image_path}, Labels: {candidate_labels})...")

        image_bytes = load_image_bytes(input_image_path)
        if not image_bytes:
            print(f"Skipping sample {i+1} due to image loading error.")
            continue # Skip if image can't be loaded

        # Prepare payload for API - specific client method handles parameters
        payload = {'parameters': {'candidate_labels': candidate_labels}}

        # Call the API
        api_result = invoke_inference_api(
            MODEL_ID,
            data=image_bytes,
            json_data=payload, # Pass labels via json_data for the specific client method
            task="zero-shot-image-classification"
        )

        if api_result:
            # API typically returns a list of {'label': '...', 'score': ...}
            try:
                # Ensure result is serializable
                predictions_json = json.dumps(api_result)
                data.append({
                    'input_image_path': input_image_path,
                    'candidate_labels': json.dumps(candidate_labels), # Store the candidates used
                    'predictions': predictions_json
                })
            except Exception as e:
                 print(f"Warning: Could not process or serialize API result for {input_image_path}: {e}")
                 print(f"API Result: {api_result}")
        else:
            print(f"Warning: API call failed for {input_image_path}. Skipping sample.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No zero-shot image classification data was generated.")

if __name__ == "__main__":
    print("Starting Zero-Shot Image Classification data generation using API...")
    # Ensure output directory exists (handled in config_vision.py)
    # os.makedirs(OUTPUT_DIR, exist_ok=True) # Already handled in config
    generate_zero_shot_image_classification_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nAPI-based data generation process finished.")
