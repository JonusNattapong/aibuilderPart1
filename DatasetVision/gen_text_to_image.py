import os
import pandas as pd
import json
from config_vision import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    TEXT_TO_IMAGE_MODEL_ID as MODEL_ID,
    TEXT_TO_IMAGE_FILENAME as FILENAME,
    TEXT_TO_IMAGE_PROMPTS as PROMPTS
)
from vision_utils import invoke_inference_api, save_image

def generate_text_to_image_data(num_samples, output_dir):
    """Generates Text-to-Image data using HF Inference API."""
    print(f"\nGenerating {num_samples} text-to-image samples via API ({MODEL_ID})...")
    data = []
    if not PROMPTS:
        print("Warning: No prompts configured in config_vision.py. Cannot generate data.")
        return

    num_to_generate = min(num_samples, len(PROMPTS))

    for i in range(num_to_generate):
        prompt = PROMPTS[i]
        print(f"Processing sample {i + 1}/{num_to_generate} (Prompt: '{prompt}')...")

        # Prepare JSON payload for the API
        payload = {"inputs": prompt}

        # Call the API
        # The client.text_to_image method directly handles this
        api_result = invoke_inference_api(MODEL_ID, json_data=payload, task="text-to-image")

        if api_result:
            # API returns PIL Image object or bytes
            image_filename_prefix = f"t2i_{i}"
            saved_path = save_image(api_result, "text_to_image", image_filename_prefix)

            if saved_path:
                data.append({
                    'prompt': prompt,
                    'generated_image_path': saved_path
                })
            else:
                print(f"Warning: Failed to save image for prompt: '{prompt}'. Skipping sample.")
        else:
            print(f"Warning: API call failed for prompt: '{prompt}'. Skipping sample.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No text-to-image data was generated.")

if __name__ == "__main__":
    print("Starting Text-to-Image data generation using API...")
    # Ensure output directory exists (handled in config_vision.py)
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_text_to_image_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nAPI-based data generation process finished.")
