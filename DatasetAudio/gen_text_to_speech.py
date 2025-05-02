import os
import pandas as pd
import uuid
from config_audio import (
    OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    TTS_MODEL_ID as MODEL_ID,
    TTS_FILENAME as FILENAME,
    TTS_INPUT_TEXTS as INPUT_TEXTS,
    BASE_PATH
)
from audio_utils import invoke_inference_api, save_audio

def generate_tts_data(num_samples, output_dir):
    """Generates Text-to-Speech data using HF Inference API."""
    print(f"\nGenerating {num_samples} Text-to-Speech samples via API ({MODEL_ID})...")
    data = []
    if not INPUT_TEXTS:
        print("Warning: No input texts configured in config_audio.py. Cannot generate data.")
        return

    num_to_generate = min(num_samples, len(INPUT_TEXTS))

    for i in range(num_to_generate):
        input_text = INPUT_TEXTS[i]
        print(f"Processing sample {i + 1}/{num_to_generate} (Input Text: '{input_text[:50]}...')...")

        # Call the API
        audio_bytes = invoke_inference_api(MODEL_ID, text=input_text, task="text-to-speech")

        if audio_bytes:
            # Save the generated audio
            # Use a unique ID or index for the filename to avoid collisions
            file_prefix = f"tts_{uuid.uuid4()}"
            relative_audio_path = save_audio(audio_bytes, "tts", file_prefix, extension=".wav") # Assume WAV output

            if relative_audio_path:
                data.append({
                    'id': str(uuid.uuid4()),
                    'input_text': input_text,
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
        print(f"Successfully generated and saved {len(data)} TTS samples to {output_path}")
    else:
        print("No Text-to-Speech data was generated.")

if __name__ == "__main__":
    print("Starting Text-to-Speech data generation using API...")
    generate_tts_data(NUM_SAMPLES_PER_TASK, OUTPUT_DIR)
    print("\nTTS data generation process finished.")
