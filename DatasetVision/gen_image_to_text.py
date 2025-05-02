import os
import pandas as pd
import json
from config_vision import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    IMAGE_TO_TEXT_MODEL_ID as MODEL_ID,
    IMAGE_TO_TEXT_FILENAME as FILENAME,
    IMAGE_TO_TEXT_INPUT_IMAGES as INPUT_IMAGES
)
from vision_utils import invoke_inference_api, load_image_bytes

def generate_image_to_text_data(num_samples, output_dir):
    """Generates Image-to-Text (Captioning) data using HF Inference API."""
    print(f"\nGenerating {num_samples} image-to-text samples via API ({MODEL_ID})...")
    data = []
    if not INPUT_IMAGES:
        print("Warning: No input images configured in config_vision.py. Cannot generate data.")
        return

    num_to_generate = min(num_samples, len(INPUT_IMAGES))

    for i in range(num_to_generate):
        input_image_path = INPUT_IMAGES[i]
        print(f"Processing sample {i + 1}/{num_to_generate} (Input: {input_image_path})...")

        image_bytes = load_image_bytes(input_image_path)
        if not image_bytes:
            print(f"Skipping sample {i+1} due to image loading error.")
            continue # Skip if image can't be loaded

        # Call the API
        api_result = invoke_inference_api(MODEL_ID, data=image_bytes, task="image-to-text")

        if api_result:
            # API typically returns a string caption, sometimes in a list/dict
            caption = ""
            if isinstance(api_result, list) and len(api_result) > 0 and isinstance(api_result[0], dict) and 'generated_text' in api_result[0]:
                 caption = api_result[0]['generated_text']
            elif isinstance(api_result, str):
                 caption = api_result
            else:
                 print(f"Warning: Unexpected API result format for {input_image_path}: {api_result}")

            if caption:
                data.append({
                    'input_image_path': input_image_path,
                    'generated_caption': caption
                })
            else:
                 print(f"Warning: Could not extract caption from API result for {input_image_path}.")

        else:
            print(f"Warning: API call failed for {input_image_path}. Skipping sample.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No image-to-text data was generated.")

if __name__ == "__main__":
    print("Starting Image-to-Text data generation using API...")
    # Ensure output directory exists (handled in config_vision.py)
    # os.makedirs(OUTPUT_DIR, exist_ok=True) # Already handled in config
    generate_image_to_text_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nAPI-based data generation process finished.")
