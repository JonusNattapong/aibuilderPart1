import os
import pandas as pd
import uuid
from transformers import AutoProcessor, AutoModelForCausalLM # Example for GIT
# Or potentially other classes depending on the model architecture
from config_multimodal import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK, DEVICE, BASE_PATH,
    VIDEO_CAPTIONING_MODEL_ID as MODEL_ID,
    VIDEO_CAPTIONING_FILENAME as FILENAME,
    VIDEO_CAPTIONING_INPUT_VIDEOS as INPUT_VIDEOS
)
from multimodal_utils import load_model_and_processor, load_video_frames, generate_video_caption

# --- Choose the correct Processor and Model classes based on VIDEO_CAPTIONING_MODEL_ID ---
# Adjust these lines if you change VIDEO_CAPTIONING_MODEL_ID in the config
# Example for microsoft/git-base-vatex
ProcessorClass = AutoProcessor
ModelClass = AutoModelForCausalLM # GIT uses causal LM head for generation
# ---

def generate_video_captioning_dataset(num_samples, output_dir):
    """Generates Video Captioning data using a local model."""
    print(f"\nGenerating {num_samples} Video Captioning samples locally ({MODEL_ID} on {DEVICE})...")
    print("Note: This requires the 'decord' library (pip install decord).")
    data = []

    if not INPUT_VIDEOS:
        print("Error: Input videos are not configured correctly in config_multimodal.py.")
        return
    if len(INPUT_VIDEOS) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {len(INPUT_VIDEOS)} videos configured.")
        num_samples = len(INPUT_VIDEOS)

    # Load model and processor
    model, processor = load_model_and_processor(MODEL_ID, ModelClass, ProcessorClass)
    if not model or not processor:
        print("Failed to load model or processor. Aborting Video Captioning generation.")
        return

    for i in range(num_samples):
        video_path_rel = INPUT_VIDEOS[i]
        print(f"Processing sample {i + 1}/{num_samples} (Video: {video_path_rel})...")

        # Load video frames
        # Adjust num_frames based on model requirements if necessary
        video_frames = load_video_frames(video_path_rel, num_frames=16)
        if video_frames is None:
            print(f"Skipping sample {i+1} due to video loading/processing error.")
            continue

        # Generate caption
        caption = generate_video_caption(model, processor, video_frames)

        if caption is not None:
            data.append({
                'id': str(uuid.uuid4()),
                'video_path': video_path_rel.replace('\\', '/'),
                'generated_caption': caption
            })
        else:
            print(f"Warning: Failed to generate caption for sample {i+1}. Skipping.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} Video Captioning samples to {output_path}")
    else:
        print("No Video Captioning data was generated.")

if __name__ == "__main__":
    print("Starting Video Captioning data generation using local model...")
    generate_video_captioning_dataset(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nVideo Captioning data generation process finished.")
