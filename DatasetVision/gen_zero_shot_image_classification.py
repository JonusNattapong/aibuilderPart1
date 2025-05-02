import os
import pandas as pd
import random
import json

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataOutput')
DATASET_VISION_DIR = os.path.dirname(__file__)
FILENAME = "generated_zero_shot_image_classification.csv"
NUM_SAMPLES = 10 # Placeholder number of samples
ALL_CLASSES = ['cat', 'dog', 'bird', 'car', 'tree', 'house', 'person'] # Placeholder classes

def generate_zero_shot_image_classification_data(num_samples, output_dir):
    """Generates placeholder Zero-Shot Image Classification data."""
    print(f"\nGenerating {num_samples} placeholder zero-shot image classification samples...")
    # Placeholder data structure: image path, candidate labels (JSON), expected label
    data = []
    for i in range(num_samples):
        candidate_labels = random.sample(ALL_CLASSES, k=random.randint(2, 5))
        expected_label = random.choice(candidate_labels) # Simplified assumption
        data.append({
            'image_path': f'images/zeroshot/img_{i}.jpg',
            'candidate_labels': json.dumps(candidate_labels),
            'expected_label': expected_label
        })

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} placeholder samples to {output_path}")
    else:
        print("No placeholder zero-shot image classification data was generated.")

if __name__ == "__main__":
    print("Starting placeholder Zero-Shot Image Classification data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_VISION_DIR, exist_ok=True)
    generate_zero_shot_image_classification_data(NUM_SAMPLES, OUTPUT_DIR)
    print("\nPlaceholder data generation process finished.")
