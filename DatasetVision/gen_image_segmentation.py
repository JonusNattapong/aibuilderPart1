import os
import pandas as pd
import json
from config_vision import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    IMAGE_SEGMENTATION_MODEL_ID as MODEL_ID,
    IMAGE_SEGMENTATION_FILENAME as FILENAME,
    IMAGE_SEGMENTATION_INPUT_IMAGES as INPUT_IMAGES
)
from vision_utils import invoke_inference_api, load_image_bytes, save_image # save_image might be needed if API returns image masks

def generate_image_segmentation_data(num_samples, output_dir):
    """Generates Image Segmentation data using HF Inference API."""
    print(f"\nGenerating {num_samples} image segmentation samples via API ({MODEL_ID})...")
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
        api_result = invoke_inference_api(MODEL_ID, data=image_bytes, task="image-segmentation")

        if api_result:
            # API likely returns a list of dicts with 'mask', 'label', 'score'
            # The 'mask' might be a PIL Image or other format.
            # For simplicity here, we store the JSON representation.
            # Handling and saving individual mask images would require parsing api_result.
            try:
                # Attempt to serialize the result directly
                segmentation_json = json.dumps(str(api_result)) # Convert complex objects to string if direct JSON fails
                data.append({
                    'input_image_path': input_image_path,
                    'segmentation_results': segmentation_json
                    # Optionally, add code here to process api_result,
                    # save individual mask images using save_image,
                    # and store their paths instead of the raw JSON.
                })
            except Exception as e:
                 print(f"Warning: Could not process or serialize API result for {input_image_path}: {e}")
                 print(f"API Result (raw): {api_result}")
        else:
            print(f"Warning: API call failed for {input_image_path}. Skipping sample.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No image segmentation data was generated.")

if __name__ == "__main__":
    print("Starting Image Segmentation data generation using API...")
    # Ensure output directory exists (handled in config_vision.py)
    # os.makedirs(OUTPUT_DIR, exist_ok=True) # Already handled in config
    generate_image_segmentation_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nAPI-based data generation process finished.")
