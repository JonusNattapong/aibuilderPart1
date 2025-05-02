import os
import pandas as pd
import random
import json

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataOutput')
DATASET_VISION_DIR = os.path.dirname(__file__)
FILENAME = "generated_image_feature_extraction.csv"
NUM_SAMPLES = 10 # Placeholder number of samples
FEATURE_DIM = 128 # Placeholder feature dimension

def generate_image_feature_extraction_data(num_samples, output_dir):
    """Generates placeholder Image Feature Extraction data."""
    print(f"\nGenerating {num_samples} placeholder image feature extraction samples...")
    # Placeholder data structure: image path and feature vector (as JSON string)
    data = []
    for i in range(num_samples):
        # Generate a random feature vector
        feature_vector = [random.random() for _ in range(FEATURE_DIM)]
        data.append({
            'image_path': f'images/features/img_{i}.jpg',
            'feature_vector': json.dumps(feature_vector)
        })

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} placeholder samples to {output_path}")
    else:
        print("No placeholder image feature extraction data was generated.")

if __name__ == "__main__":
    print("Starting placeholder Image Feature Extraction data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_VISION_DIR, exist_ok=True)
    generate_image_feature_extraction_data(NUM_SAMPLES, OUTPUT_DIR)
    print("\nPlaceholder data generation process finished.")
