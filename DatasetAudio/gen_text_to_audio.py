import os
import pandas as pd
import uuid
from config_audio import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    TEXT_TO_AUDIO_MODEL_ID as MODEL_ID,
    TEXT_TO_AUDIO_FILENAME as FILENAME,
    TEXT_TO_AUDIO_PROMPTS as INPUT_PROMPTS,
    BASE_PATH
)
from audio_utils import invoke_inference_api, save_audio

def generate_text_to_audio_data(num_samples, output_dir):
    """Generates Text-to-Audio (Sound Generation) data using HF Inference API (Experimental)."""
    print(f"\nGenerating {num_samples} Text-to-Audio samples via API ({MODEL_ID})... (Experimental)")
    data = []
    if not INPUT_PROMPTS:
        print("Warning: No input prompts configured in config_audio.py. Cannot generate data.")
        return

    num_to_generate = min(num_samples, len(INPUT_PROMPTS))

    for i in range(num_to_generate):
        input_prompt = INPUT_PROMPTS[i]
        print(f"Processing sample {i + 1}/{num_to_generate} (Input Prompt: '{input_prompt[:50]}...')...")

        # Call the API - Task name might need adjustment based on actual model endpoint
        audio_bytes = invoke_inference_api(MODEL_ID, text=input_prompt, task="text-to-audio")

        if audio_bytes:
            # Save the generated audio
            file_prefix = f"t2a_{uuid.uuid4()}"
            # Determine appropriate extension (might be .wav, .mp3, etc. depending on model)
            extension = ".wav" # Assume WAV for now
            relative_audio_path = save_audio(audio_bytes, "text_to_audio", file_prefix, extension=extension)

            if relative_audio_path:
                data.append({
                    'id': str(uuid.uuid4()),
                    'input_prompt': input_prompt,
                    'generated_audio_path': relative_audio_path
                })
            else:
                print(f"Warning: Failed to save generated audio for sample {i+1}.")
        else:
            print(f"Warning: API call failed for sample {i+1}. Skipping.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} Text-to-Audio samples to {output_path}")
    else:
        print("No Text-to-Audio data was generated.")

if __name__ == "__main__":
    print("Starting Text-to-Audio data generation using API (Experimental)...")
    generate_text_to_audio_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nText-to-Audio data generation process finished.")
