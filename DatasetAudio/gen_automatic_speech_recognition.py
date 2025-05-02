import os
import pandas as pd
import uuid
from config_audio import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    ASR_MODEL_ID as MODEL_ID,
    ASR_FILENAME as FILENAME,
    ASR_INPUT_AUDIO as INPUT_AUDIO,
    BASE_PATH
)
from audio_utils import invoke_inference_api, load_audio_bytes

def generate_asr_data(num_samples, output_dir):
    """Generates Automatic Speech Recognition data using HF Inference API."""
    print(f"\nGenerating {num_samples} ASR samples via API ({MODEL_ID})...")
    data = []
    if not INPUT_AUDIO:
        print("Warning: No input audio files configured in config_audio.py. Cannot generate data.")
        return

    num_to_generate = min(num_samples, len(INPUT_AUDIO))

    for i in range(num_to_generate):
        input_audio_path_rel = INPUT_AUDIO[i] # Path relative to project root or placeholder dir
        input_audio_path_abs = os.path.join(BASE_PATH, input_audio_path_rel) # Ensure absolute path for loading
        print(f"Processing sample {i + 1}/{num_to_generate} (Input Audio: {input_audio_path_rel})...")

        audio_bytes = load_audio_bytes(input_audio_path_abs)
        if not audio_bytes:
            print(f"Skipping sample {i+1} due to audio loading error.")
            continue # Skip if audio can't be loaded

        # Call the API
        transcription = invoke_inference_api(MODEL_ID, data=audio_bytes, task="automatic-speech-recognition")

        if transcription is not None: # API might return empty string for silence
             # Ensure transcription is a string
            transcription_text = str(transcription)
            data.append({
                'id': str(uuid.uuid4()),
                'input_audio_path': input_audio_path_rel.replace('\\', '/'), # Use relative path in CSV
                'generated_transcription': transcription_text
            })
        else:
            print(f"Warning: API call failed or returned None for sample {i+1}. Skipping.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} ASR samples to {output_path}")
    else:
        print("No ASR data was generated.")

if __name__ == "__main__":
    print("Starting Automatic Speech Recognition data generation using API...")
    generate_asr_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nASR data generation process finished.")
