import os
import torch
from PIL import Image
from config_multimodal import BASE_PATH, DEVICE

# --- Model Loading Cache ---
_model_cache = {}
_processor_cache = {}

def load_model_and_processor(model_id, model_class, processor_class):
    """Loads a model and processor, caching them for reuse."""
    global _model_cache, _processor_cache

    if model_id in _model_cache and model_id in _processor_cache:
        print(f"Using cached model and processor for {model_id}")
        return _model_cache[model_id], _processor_cache[model_id]

    print(f"Loading model and processor for {model_id}...")
    try:
        processor = processor_class.from_pretrained(model_id)
        model = model_class.from_pretrained(model_id)
        model.to(DEVICE)
        model.eval()
        _processor_cache[model_id] = processor
        _model_cache[model_id] = model
        print(f"Successfully loaded {model_id} to {DEVICE}")
        return model, processor
    except Exception as e:
        print(f"Error loading model/processor {model_id}: {e}")
        return None, None

def load_image(image_path):
    """Loads an image file using PIL."""
    full_path = os.path.join(BASE_PATH, image_path) if not os.path.isabs(image_path) else image_path
    if not os.path.exists(full_path):
        print(f"Error: Image file not found at {full_path}")
        return None
    try:
        img = Image.open(full_path).convert("RGB")
        print(f"Loaded image: {full_path}")
        return img
    except Exception as e:
        print(f"Error loading image {full_path}: {e}")
        return None

def load_video_frames(video_path, num_frames=16):
    """Loads frames from a video file using decord."""
    # Requires decord: pip install decord
    try:
        import decord
        from decord import VideoReader
    except ImportError:
        print("Error: `decord` library is required for video processing. Please install it: pip install decord")
        return None

    full_path = os.path.join(BASE_PATH, video_path) if not os.path.isabs(video_path) else video_path
    if not os.path.exists(full_path):
        print(f"Error: Video file not found at {full_path}")
        return None

    try:
        vr = VideoReader(full_path, ctx=decord.cpu(0)) # Load on CPU first
        total_frames = len(vr)
        # Select frames evenly spaced throughout the video
        indices = torch.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy() # Get frames as numpy array
        print(f"Loaded {len(frames)} frames from video: {full_path}")
        return list(frames) # Return list of numpy arrays (frames)
    except Exception as e:
        print(f"Error processing video {full_path} with decord: {e}")
        return None

# --- Specific Task Generation Functions ---

def generate_vqa_answer(model, processor, image, question):
    """Generates an answer for a VQA task."""
    try:
        # Prepare inputs: process image and question text
        inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE)

        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=50) # Adjust max_length as needed

        # Decode the generated answer
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return answer
    except Exception as e:
        print(f"Error during VQA generation: {e}")
        return None

def generate_video_caption(model, processor, video_frames):
    """Generates a caption for a video using loaded frames."""
    # This function's implementation depends heavily on the specific model architecture.
    # Example for GIT model (microsoft/git-base-vatex)
    try:
        # Process video frames (might need specific frame selection/sampling)
        # Assuming processor handles list of frames or pixel_values directly
        pixel_values = processor(images=video_frames, return_tensors="pt").pixel_values.to(DEVICE)

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50) # Adjust max_length

        # Decode the caption
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption
    except Exception as e:
        print(f"Error during video captioning generation: {e}")
        # Print more details if needed
        # import traceback
        # traceback.print_exc()
        return None
