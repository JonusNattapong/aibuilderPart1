import os
import pandas as pd
import random
import json

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataOutput')
DATASET_VISION_DIR = os.path.dirname(__file__)
FILENAME = "generated_zero_shot_object_detection.csv"
NUM_SAMPLES = 10 # Placeholder number of samples
ALL_CLASSES = ['car', 'person', 'bicycle', 'dog', 'backpack', 'umbrella'] # Placeholder classes

def generate_zero_shot_object_detection_data(num_samples, output_dir):
    """Generates placeholder Zero-Shot Object Detection data."""
    print(f"\nGenerating {num_samples} placeholder zero-shot object detection samples...")
    # Placeholder data structure: image path, candidate labels (JSON), expected annotations (JSON)
    data = []
    for i in range(num_samples):
        candidate_labels = random.sample(ALL_CLASSES, k=random.randint(2, 5))
        num_objects = random.randint(1, 3)
        bboxes = []
        for _ in range(num_objects):
            x_min = random.randint(10, 50)
            y_min = random.randint(10, 50)
            width = random.randint(20, 100)
            height = random.randint(20, 100)
            # In zero-shot, the model detects based on candidate labels
            # For placeholder, we just assign one of the candidates
            label = random.choice(candidate_labels)
            bboxes.append({
                'label': label,
                'bbox': [x_min, y_min, x_min + width, y_min + height] # [xmin, ymin, xmax, ymax]
            })
        data.append({
            'image_path': f'images/zeroshot_det/img_{i}.jpg',
            'candidate_labels': json.dumps(candidate_labels),
            'expected_annotations': json.dumps(bboxes) # Ground truth for evaluation
        })

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} placeholder samples to {output_path}")
    else:
        print("No placeholder zero-shot object detection data was generated.")

if __name__ == "__main__":
    print("Starting placeholder Zero-Shot Object Detection data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_VISION_DIR, exist_ok=True)
    generate_zero_shot_object_detection_data(NUM_SAMPLES, OUTPUT_DIR)
    print("\nPlaceholder data generation process finished.")
