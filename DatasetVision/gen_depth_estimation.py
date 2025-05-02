import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataOutput')
DATASET_VISION_DIR = os.path.dirname(__file__)
FILENAME = "generated_depth_estimation.csv"
NUM_SAMPLES = 10 # Placeholder number of samples

def generate_depth_estimation_data(num_samples, output_dir):
    """Generates placeholder Depth Estimation data."""
    print(f"\nGenerating {num_samples} placeholder depth estimation samples...")
    # Placeholder data structure: image path and corresponding depth map path
    data = [
        {'image_path': f'images/img_{i}.jpg', 'depth_map_path': f'depth_maps/depth_{i}.png'}
        for i in range(num_samples)
    ]

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} placeholder samples to {output_path}")
    else:
        print("No placeholder depth estimation data was generated.")

if __name__ == "__main__":
    print("Starting placeholder Depth Estimation data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_VISION_DIR, exist_ok=True)
    generate_depth_estimation_data(NUM_SAMPLES, OUTPUT_DIR)
    print("\nPlaceholder data generation process finished.")
