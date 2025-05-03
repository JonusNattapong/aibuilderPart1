import os
import time
import requests
import json
from huggingface_hub import InferenceClient
from config_audio import HF_API_TOKEN, MAX_RETRIES, RETRY_DELAY, GENERATED_MEDIA_DIR, BASE_PATH
import io
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv() # Added

# Initialize Inference Client (if token is provided)
client = None
if HF_API_TOKEN:
    print("Initializing Hugging Face Inference Client for Audio...")
    client = InferenceClient(token=HF_API_TOKEN)
else:
    print("Warning: HF_TOKEN environment variable not set. Inference API calls will likely fail.")
    print("For API usage, set the HF_TOKEN environment variable with your Hugging Face API token.")

def load_audio_bytes(audio_path):
    """Loads an audio file into bytes."""
    # Ensure the path is absolute or relative to BASE_PATH
    if not os.path.isabs(audio_path):
        full_path = os.path.join(BASE_PATH, audio_path)
    else:
        full_path = audio_path

    if not os.path.exists(full_path):
        print(f"Error: Audio file not found at {full_path}")
        return None
    try:
        with open(full_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading audio file {full_path}: {e}")
        return None

def save_audio(audio_bytes, task_name, filename_prefix, extension=".wav"):
    """Saves audio bytes to a file in the generated media directory."""
    if not audio_bytes:
        print("Error: No audio bytes provided to save.")
        return None

    # Create task-specific subdirectory if desired (optional)
    # task_media_dir = os.path.join(GENERATED_MEDIA_DIR, task_name)
    # os.makedirs(task_media_dir, exist_ok=True)
    # output_filename = f"{filename_prefix}{extension}"
    # output_path = os.path.join(task_media_dir, output_filename)

    # Save directly in GENERATED_MEDIA_DIR for simplicity
    output_filename = f"{task_name}_{filename_prefix}{extension}"
    output_path = os.path.join(GENERATED_MEDIA_DIR, output_filename)

    try:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Successfully saved audio to {output_path}")
        # Return relative path for CSV
        relative_path = os.path.relpath(output_path, BASE_PATH).replace('\\', '/')
        return relative_path
    except Exception as e:
        print(f"Error saving audio to {output_path}: {e}")
        return None

def invoke_inference_api(model_id, data=None, text=None, task=None):
    """Invokes the Hugging Face Inference API for audio tasks with retries."""
    if not client:
        print(f"Skipping API call for {model_id} as client is not initialized.")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Calling model {model_id} (Task: {task}) via Inference API...")

            if task == "text-to-speech":
                # Returns audio bytes
                response = client.text_to_speech(text=text, model=model_id)
                return response # Should be bytes
            elif task == "automatic-speech-recognition":
                 # Expects audio bytes in 'data'
                response = client.automatic_speech_recognition(audio=data, model=model_id)
                # Returns dict like {'text': '...'}
                return response.get('text') if isinstance(response, dict) else response
            elif task == "audio-classification":
                 # Expects audio bytes in 'data'
                response = client.audio_classification(audio=data, model=model_id)
                # Returns list of dicts like [{'score': 0.99, 'label': 'Music'}]
                return response
            elif task == "text-to-audio":
                 # Expects text prompt
                 # This task might not be standard, depends on model endpoint
                 # Assuming it returns audio bytes similar to TTS for now
                 # May need custom request structure if not directly supported by client
                 print(f"Warning: 'text-to-audio' task support via client is experimental/model-dependent.")
                 # Example using generic post if needed:
                 # response = client.post(json={"inputs": text}, model=model_id) # Adjust payload as needed
                 # For now, assume a hypothetical direct method:
                 response = client.text_to_audio(text=text, model=model_id) # Hypothetical
                 return response # Should be bytes
            elif task == "audio-to-audio":
                 # Expects audio bytes in 'data'
                 # Highly model specific, might return audio bytes or structured data
                 print(f"Warning: 'audio-to-audio' task support via client is experimental/model-dependent.")
                 # Example using generic post if needed:
                 # response = client.post(data=data, model=model_id, task="audio-to-audio") # Adjust payload/task
                 # For now, assume a hypothetical direct method:
                 response = client.audio_to_audio(audio=data, model=model_id) # Hypothetical
                 return response # Could be bytes or dict/list
            elif task == "voice-activity-detection":
                 # Expects audio bytes in 'data'
                 # Often returns timestamps or segments, format varies
                 print(f"Warning: 'voice-activity-detection' task support via client is experimental/model-dependent.")
                 # Example using generic post if needed:
                 # response = client.post(data=data, model=model_id, task="voice-activity-detection") # Adjust payload/task
                 # For now, assume a hypothetical direct method:
                 response = client.voice_activity_detection(audio=data, model=model_id) # Hypothetical
                 return response # Likely dict/list

            else:
                print(f"Error: Unknown audio task '{task}'")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Network error during API call (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            # Catch other potential errors from the client or API response processing
            print(f"Error during API call or processing (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            # Specific handling for common API errors if needed
            # if "Model is currently loading" in str(e):
            #     print("Model is loading, retrying...")
            #     time.sleep(RETRY_DELAY * 2) # Longer delay for loading models
            # else:
            #     time.sleep(RETRY_DELAY)
            time.sleep(RETRY_DELAY)


    print(f"API call failed after {MAX_RETRIES} attempts for model {model_id}.")
    return None
