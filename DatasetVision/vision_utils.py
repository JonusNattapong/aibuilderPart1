import os
import time
import requests
import json
from huggingface_hub import InferenceClient, hf_hub_download
from PIL import Image
import io
from config_vision import HF_API_TOKEN, MAX_RETRIES, RETRY_DELAY, GENERATED_MEDIA_DIR

# Initialize Inference Client (if token is provided)
client = None
if HF_API_TOKEN:
    print("Initializing Hugging Face Inference Client...")
    client = InferenceClient(token=HF_API_TOKEN)
else:
    print("Warning: HF_TOKEN environment variable not set. Inference API calls will likely fail.")
    print("For API usage, set the HF_TOKEN environment variable with your Hugging Face API token.")

def invoke_inference_api(model_id, data=None, json_data=None, task=None):
    """Invokes the Hugging Face Inference API with retries."""
    if not client:
        print(f"Skipping API call for {model_id} as client is not initialized.")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Calling model {model_id} via Inference API...")
            if task == "text-to-image":
                # Text-to-image often returns raw image bytes
                response = client.text_to_image(prompt=json_data['inputs'], model=model_id)
                return response # Should be PIL Image or bytes
            elif task == "image-to-text":
                 # Assumes data is image bytes
                response = client.image_to_text(image=data, model=model_id)
                return response # Should be string caption
            elif task == "image-classification":
                 # Assumes data is image bytes
                response = client.image_classification(image=data, model=model_id)
                return response # Should be list of dicts (label, score)
            elif task == "object-detection":
                 # Assumes data is image bytes
                response = client.object_detection(image=data, model=model_id)
                return response # Should be list of dicts (box, label, score)
            elif task == "depth-estimation":
                 # Assumes data is image bytes
                response = client.depth_estimation(image=data, model=model_id)
                return response # Should be PIL Image (depth map)
            elif task == "image-segmentation":
                 # Assumes data is image bytes
                response = client.image_segmentation(image=data, model=model_id)
                return response # Should be list of dicts (mask, label, score)
            elif task == "zero-shot-image-classification":
                 # Assumes data is image bytes, json_data contains parameters
                response = client.zero_shot_image_classification(
                    image=data,
                    candidate_labels=json_data['parameters']['candidate_labels'],
                    model=model_id
                )
                return response # Should be list of dicts (label, score)
            else:
                # Generic POST request for other tasks (might need adjustments)
                headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
                api_url = f"https://api-inference.huggingface.co/models/{model_id}"
                payload = json_data if json_data else {}
                files = {'file': ('image.jpg', data, 'image/jpeg')} if data else None

                response = requests.post(api_url, headers=headers, json=payload, files=files)
                response.raise_for_status() # Raise an exception for bad status codes

                # Try to parse JSON, otherwise return raw content
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return response.content # Could be image bytes or other format

        except requests.exceptions.RequestException as e:
            print(f"Error during API call (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if isinstance(e, requests.exceptions.HTTPError):
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
                # Specific handling for common errors
                if e.response.status_code == 429: # Rate limited
                    print("Rate limited. Waiting longer...")
                    time.sleep(RETRY_DELAY * 5) # Wait longer for rate limits
                elif e.response.status_code >= 500: # Server error
                    print("Server error. Retrying...")
                    time.sleep(RETRY_DELAY)
                elif e.response.status_code == 400: # Bad request (often model loading)
                    print("Bad request. Model might be loading or invalid input. Waiting...")
                    time.sleep(RETRY_DELAY * 2)
                else:
                    # Don't retry for other client errors (e.g., 401 Unauthorized, 404 Not Found)
                    print("Non-retryable client error. Aborting.")
                    return None
            else: # Network errors
                print("Network error. Retrying...")
                time.sleep(RETRY_DELAY)
        except Exception as e:
            # Catch other potential errors from client.<task> methods
            print(f"Unexpected error during API call (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)

    print(f"API call failed after {MAX_RETRIES} attempts for model {model_id}.")
    return None

def save_image(image_data, task_name, filename_prefix, extension=".png"):
    """Saves image data (PIL Image or bytes) to a file."""
    try:
        task_dir = os.path.join(GENERATED_MEDIA_DIR, task_name)
        os.makedirs(task_dir, exist_ok=True)
        filepath = os.path.join(task_dir, f"{filename_prefix}{extension}")

        if isinstance(image_data, Image.Image):
            image_data.save(filepath)
        elif isinstance(image_data, bytes):
            with open(filepath, "wb") as f:
                f.write(image_data)
        else:
            print(f"Warning: Unsupported image data type: {type(image_data)}. Cannot save.")
            return None

        print(f"Saved image to {filepath}")
        # Return relative path for CSV
        return os.path.join('generated_media', task_name, f"{filename_prefix}{extension}").replace('\\', '/')
    except Exception as e:
        print(f"Error saving image {filename_prefix}: {e}")
        return None

def load_image_bytes(image_path):
    """Loads an image from a path and returns its bytes. Returns None if path is invalid or placeholder."""
    # Basic check for placeholder paths
    if "placeholder_" in image_path or not os.path.exists(image_path):
        print(f"Warning: Input image path '{image_path}' is a placeholder or does not exist. Skipping load.")
        return None
    try:
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img_format = img.format if img.format else 'JPEG' # Default to JPEG if format is None
            img.save(img_byte_arr, format=img_format)
            img_byte_arr = img_byte_arr.getvalue()
            return img_byte_arr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# --- Local Model Utilities (Placeholder) ---
# Add functions here later to load and run local models using transformers/diffusers
# Example structure:
# def load_local_model(model_id, task):
#     pass
# def run_local_inference(model, tokenizer_or_processor, input_data, task):
#     pass
