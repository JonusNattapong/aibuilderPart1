import os

# --- General Configuration ---
# Determine the base path (assuming this script is in the 'DatasetMultimodal' folder)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_PATH, 'DataOutput')
NUM_SAMPLES_PER_TASK = 3 # Keep low for local testing
DEVICE = "cuda" # Use "cuda" if GPU is available and configured, otherwise "cpu"

# --- Input Data Paths ---
# Assumes placeholder directories exist at the project root
PLACEHOLDER_IMAGE_DIR = os.path.join(BASE_PATH, 'placeholder_images')
PLACEHOLDER_VIDEO_DIR = os.path.join(BASE_PATH, 'placeholder_videos')

# --- Task: Visual Question Answering (VQA) ---
VQA_MODEL_ID = "Salesforce/blip-vqa-base" # Example VQA model
# VQA_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa" # Alternative VQA model
VQA_FILENAME = "generated_vqa_local.csv"
VQA_INPUT_IMAGES = [os.path.join(PLACEHOLDER_IMAGE_DIR, f'img_{i}.jpg') for i in range(NUM_SAMPLES_PER_TASK)] # Use placeholder image paths
VQA_QUESTIONS = [
    "What is the main subject?",
    "What color is the object?",
    "Is there a person in the image?",
    "Describe the scene.",
    "What is happening here?"
][:NUM_SAMPLES_PER_TASK] # Ensure questions match the number of images

# --- Task: Video-Text-to-Text (Implemented as Video Captioning) ---
# Note: Video processing locally can be complex and require specific libraries (e.g., decord, av)
# Ensure the chosen model is suitable for captioning/description and libraries are installed.
VIDEO_CAPTIONING_MODEL_ID = "microsoft/git-base-vatex" # Example Video Captioning model
VIDEO_CAPTIONING_FILENAME = "generated_video_captioning_local.csv"
VIDEO_CAPTIONING_INPUT_VIDEOS = [os.path.join(PLACEHOLDER_VIDEO_DIR, f'vid_{i}.mp4') for i in range(NUM_SAMPLES_PER_TASK)] # Use placeholder video paths

# --- Helper to ensure directories exist ---
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Ensure placeholder dirs exist (optional, user should create them)
    os.makedirs(PLACEHOLDER_IMAGE_DIR, exist_ok=True)
    os.makedirs(PLACEHOLDER_VIDEO_DIR, exist_ok=True)

ensure_dirs()
print(f"Multimodal Configuration Loaded. Output Dir: {OUTPUT_DIR}")
print(f"Using Device: {DEVICE}")
print(f"Placeholder images expected in: {PLACEHOLDER_IMAGE_DIR}")
print(f"Placeholder videos expected in: {PLACEHOLDER_VIDEO_DIR}")
# Check if input files exist (optional check)
if not all(os.path.exists(p) for p in VQA_INPUT_IMAGES if os.path.basename(p).startswith('img_')):
     print(f"Warning: Not all placeholder images found in {PLACEHOLDER_IMAGE_DIR}. VQA generation might fail.")
if not all(os.path.exists(p) for p in VIDEO_CAPTIONING_INPUT_VIDEOS if os.path.basename(p).startswith('vid_')):
     print(f"Warning: Not all placeholder videos found in {PLACEHOLDER_VIDEO_DIR}. Video Captioning generation might fail.")

# Check for necessary libraries (optional)
try:
    import transformers
    import torch
    import PIL
    print("transformers, torch, Pillow found.")
except ImportError as e:
    print(f"Warning: Missing core library: {e}. Install with 'pip install transformers torch Pillow'")

try:
    import decord
    print("decord found (needed for some video models).")
except ImportError:
    print("Warning: `decord` library not found. Video processing might fail depending on the model.")
    print("Install it with: pip install decord")
