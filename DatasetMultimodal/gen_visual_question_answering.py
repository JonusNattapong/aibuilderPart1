import os
import pandas as pd
import uuid
from transformers import AutoProcessor, BlipForQuestionAnswering # Example model class
# from transformers import ViltProcessor, ViltForQuestionAnswering # Alternative
from config_multimodal import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK, DEVICE, BASE_PATH,
    VQA_MODEL_ID as MODEL_ID,
    VQA_FILENAME as FILENAME,
    VQA_INPUT_IMAGES as INPUT_IMAGES,
    VQA_QUESTIONS as QUESTIONS
)
from multimodal_utils import load_model_and_processor, load_image, generate_vqa_answer

# --- Choose the correct Processor and Model classes based on VQA_MODEL_ID ---
# Adjust these lines if you change VQA_MODEL_ID in the config
ProcessorClass = AutoProcessor
ModelClass = BlipForQuestionAnswering
# If using Vilt:
# ProcessorClass = ViltProcessor
# ModelClass = ViltForQuestionAnswering
# ---

def generate_vqa_dataset(num_samples, output_dir):
    """Generates Visual Question Answering data using a local model."""
    print(f"\nGenerating {num_samples} VQA samples locally ({MODEL_ID} on {DEVICE})...")
    data = []

    if not INPUT_IMAGES or not QUESTIONS or len(INPUT_IMAGES) != len(QUESTIONS):
        print("Error: Input images and questions are not configured correctly in config_multimodal.py.")
        print(f"Images: {len(INPUT_IMAGES)}, Questions: {len(QUESTIONS)}")
        return
    if len(INPUT_IMAGES) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {len(INPUT_IMAGES)} image/question pairs configured.")
        num_samples = len(INPUT_IMAGES)

    # Load model and processor
    model, processor = load_model_and_processor(MODEL_ID, ModelClass, ProcessorClass)
    if not model or not processor:
        print("Failed to load model or processor. Aborting VQA generation.")
        return

    for i in range(num_samples):
        image_path_rel = INPUT_IMAGES[i]
        question = QUESTIONS[i]
        print(f"Processing sample {i + 1}/{num_samples} (Image: {image_path_rel}, Question: '{question}')...")

        # Load image
        image = load_image(image_path_rel)
        if image is None:
            print(f"Skipping sample {i+1} due to image loading error.")
            continue

        # Generate answer
        answer = generate_vqa_answer(model, processor, image, question)

        if answer is not None:
            data.append({
                'id': str(uuid.uuid4()),
                'image_path': image_path_rel.replace('\\', '/'),
                'question': question,
                'generated_answer': answer
            })
        else:
            print(f"Warning: Failed to generate answer for sample {i+1}. Skipping.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} VQA samples to {output_path}")
    else:
        print("No VQA data was generated.")

if __name__ == "__main__":
    print("Starting Visual Question Answering data generation using local model...")
    generate_vqa_dataset(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nVQA data generation process finished.")
