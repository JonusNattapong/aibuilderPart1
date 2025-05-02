import os
import pandas as pd
import random
import json

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataOutput')
DATASET_VISION_DIR = os.path.dirname(__file__)
FILENAME = "generated_object_detection.csv"
NUM_SAMPLES = 10 # Placeholder number of samples
CLASSES = ['car', 'person', 'traffic_light'] # Placeholder classes

def generate_object_detection_data(num_samples, output_dir):
    """Generates placeholder Object Detection data."""
    print(f"\nGenerating {num_samples} placeholder object detection samples...")
    # Placeholder data structure: image path and bounding boxes (as JSON string)
    data = []
    for i in range(num_samples):
        num_objects = random.randint(1, 5)
        bboxes = []
        for _ in range(num_objects):
            x_min = random.randint(10, 50)
            y_min = random.randint(10, 50)
            width = random.randint(20, 100)
            height = random.randint(20, 100)
            bboxes.append({
                'label': random.choice(CLASSES),
                'bbox': [x_min, y_min, x_min + width, y_min + height] # [xmin, ymin, xmax, ymax]
            })
        data.append({
            'image_path': f'images/detection/img_{i}.jpg',
            'annotations': json.dumps(bboxes)
        })

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} placeholder samples to {output_path}")
    else:
        print("No placeholder object detection data was generated.")

if __name__ == "__main__":
    print("Starting placeholder Object Detection data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_VISION_DIR, exist_ok=True)
    generate_object_detection_data(NUM_SAMPLES, OUTPUT_DIR)
    print("\nPlaceholder data generation process finished.")
