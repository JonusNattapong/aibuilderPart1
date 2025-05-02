import os
import pandas as pd
import random

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataOutput')
DATASET_VISION_DIR = os.path.dirname(__file__)
FILENAME = "generated_video_classification.csv"
NUM_SAMPLES = 10 # Placeholder number of samples
CLASSES = ['playing sports', 'cooking', 'driving'] # Placeholder classes

def generate_video_classification_data(num_samples, output_dir):
    """Generates placeholder Video Classification data."""
    print(f"\nGenerating {num_samples} placeholder video classification samples...")
    # Placeholder data structure: video path and label
    data = [
        {'video_path': f'videos/classification/vid_{i}.mp4', 'label': random.choice(CLASSES)}
        for i in range(num_samples)
    ]

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} placeholder samples to {output_path}")
    else:
        print("No placeholder video classification data was generated.")

if __name__ == "__main__":
    print("Starting placeholder Video Classification data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_VISION_DIR, exist_ok=True)
    generate_video_classification_data(NUM_SAMPLES, OUTPUT_DIR)
    print("\nPlaceholder data generation process finished.")
