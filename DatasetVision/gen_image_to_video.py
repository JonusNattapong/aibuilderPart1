import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataOutput')
DATASET_VISION_DIR = os.path.dirname(__file__)
FILENAME = "generated_image_to_video.csv"
NUM_SAMPLES = 10 # Placeholder number of samples

def generate_image_to_video_data(num_samples, output_dir):
    """Generates placeholder Image-to-Video data."""
    print(f"\nGenerating {num_samples} placeholder image-to-video samples...")
    # Placeholder data structure: input image path, generated video path
    data = [
        {'input_image_path': f'images/input/img_{i}.jpg', 'generated_video_path': f'generated_videos/i2v_{i}.mp4'}
        for i in range(num_samples)
    ]

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} placeholder samples to {output_path}")
    else:
        print("No placeholder image-to-video data was generated.")

if __name__ == "__main__":
    print("Starting placeholder Image-to-Video data generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_VISION_DIR, exist_ok=True)
    generate_image_to_video_data(NUM_SAMPLES, OUTPUT_DIR)
    print("\nPlaceholder data generation process finished.")
