import os
import pandas as pd
import json
from config_vision import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    DEPTH_ESTIMATION_MODEL_ID as MODEL_ID,
    DEPTH_ESTIMATION_FILENAME as FILENAME,
    DEPTH_ESTIMATION_INPUT_IMAGES as INPUT_IMAGES
)
from vision_utils import invoke_inference_api, load_image_bytes, save_image

def generate_depth_estimation_data(num_samples, output_dir):
    """Generates Depth Estimation data using HF Inference API."""
    print(f"\nGenerating {num_samples} depth estimation samples via API ({MODEL_ID})...")
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
        api_result = invoke_inference_api(MODEL_ID, data=image_bytes, task="depth-estimation")

        if api_result:
            # API returns PIL Image object (depth map)
            depth_map_filename_prefix = f"depth_out_{i}"
            saved_path = save_image(api_result, "depth_estimation", depth_map_filename_prefix)

            if saved_path:
                data.append({
                    'input_image_path': input_image_path,
                    'generated_depth_map_path': saved_path
                })
            else:
                print(f"Warning: Failed to save depth map for {input_image_path}. Skipping sample.")
        else:
            print(f"Warning: API call failed for {input_image_path}. Skipping sample.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No depth estimation data was generated.")

if __name__ == "__main__":
    print("Starting Depth Estimation data generation using API...")
    # Ensure output directory exists (handled in config_vision.py)
    # os.makedirs(OUTPUT_DIR, exist_ok=True) # Already handled in config
    generate_depth_estimation_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nAPI-based data generation process finished.")
