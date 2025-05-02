import os

# --- General Configuration ---
# Determine the base path (assuming this script is in the 'DatasetVision' folder)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_PATH, 'DataOutput')
GENERATED_MEDIA_DIR = os.path.join(OUTPUT_DIR, 'generated_media') # Store generated images/videos here
NUM_SAMPLES_PER_TASK = 5 # Reduced for API testing
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# --- Hugging Face Inference API Configuration ---
# It's strongly recommended to set the API token via environment variable:
# export HF_TOKEN='your_hf_token'
# Or in Python before running the script: os.environ["HF_TOKEN"] = "your_hf_token"
HF_API_TOKEN = os.environ.get("HF_TOKEN") # Read from environment variable

# --- Task-Specific Model IDs (Examples for Inference API) ---
# Choose models known to work well with the Inference API for these tasks
# You might need to experiment or check the Hugging Face Hub for suitable models.

# Image Classification
IMAGE_CLASSIFICATION_MODEL_ID = "google/vit-base-patch16-224"
IMAGE_CLASSIFICATION_FILENAME = "generated_image_classification_api.csv"
# Placeholder input image paths (replace with actual paths if available)
IMAGE_CLASSIFICATION_INPUT_IMAGES = [f"placeholder_images/img_{i}.jpg" for i in range(NUM_SAMPLES_PER_TASK)]

# Object Detection
OBJECT_DETECTION_MODEL_ID = "facebook/detr-resnet-50"
OBJECT_DETECTION_FILENAME = "generated_object_detection_api.csv"
# Placeholder input image paths
OBJECT_DETECTION_INPUT_IMAGES = [f"placeholder_images/det_{i}.jpg" for i in range(NUM_SAMPLES_PER_TASK)]

# Text-to-Image
TEXT_TO_IMAGE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" # Example, might require Pro subscription or dedicated inference
TEXT_TO_IMAGE_FILENAME = "generated_text_to_image_api.csv"
TEXT_TO_IMAGE_PROMPTS = [
    "A photorealistic cat sitting on a windowsill",
    "An oil painting of a futuristic cityscape at sunset",
    "A cartoon drawing of a friendly robot waving",
    "A watercolor painting of a serene lake surrounded by mountains",
    "A high-resolution photo of a delicious-looking pizza"
]

# --- Add other task configurations here as needed ---
# Depth Estimation
DEPTH_ESTIMATION_MODEL_ID = "Intel/dpt-large" # Example
DEPTH_ESTIMATION_FILENAME = "generated_depth_estimation_api.csv"
DEPTH_ESTIMATION_INPUT_IMAGES = [f"placeholder_images/depth_in_{i}.jpg" for i in range(NUM_SAMPLES_PER_TASK)]

# Image Segmentation
IMAGE_SEGMENTATION_MODEL_ID = "facebook/detr-resnet-50-panoptic" # Example
IMAGE_SEGMENTATION_FILENAME = "generated_image_segmentation_api.csv"
IMAGE_SEGMENTATION_INPUT_IMAGES = [f"placeholder_images/seg_in_{i}.jpg" for i in range(NUM_SAMPLES_PER_TASK)]

# Image-to-Text
IMAGE_TO_TEXT_MODEL_ID = "nlpconnect/vit-gpt2-image-captioning" # Example
IMAGE_TO_TEXT_FILENAME = "generated_image_to_text_api.csv"
IMAGE_TO_TEXT_INPUT_IMAGES = [f"placeholder_images/cap_in_{i}.jpg" for i in range(NUM_SAMPLES_PER_TASK)]

# Zero-Shot Image Classification
ZERO_SHOT_IMAGE_CLASSIFICATION_MODEL_ID = "openai/clip-vit-large-patch14" # Example
ZERO_SHOT_IMAGE_CLASSIFICATION_FILENAME = "generated_zero_shot_image_classification_api.csv"
ZERO_SHOT_IMAGE_CLASSIFICATION_INPUT_IMAGES = [f"placeholder_images/zs_img_{i}.jpg" for i in range(NUM_SAMPLES_PER_TASK)]
ZERO_SHOT_IMAGE_CLASSIFICATION_CANDIDATES = [
    ["cat", "dog", "car"],
    ["tree", "house", "person"],
    ["flower", "bird", "sky"],
    ["computer", "desk", "chair"],
    ["food", "plate", "table"]
]

# --- Helper to ensure directories exist ---
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GENERATED_MEDIA_DIR, exist_ok=True)
    # Create subdirs for specific tasks if needed
    os.makedirs(os.path.join(GENERATED_MEDIA_DIR, 'text_to_image'), exist_ok=True)
    os.makedirs(os.path.join(GENERATED_MEDIA_DIR, 'depth_estimation'), exist_ok=True)
    os.makedirs(os.path.join(GENERATED_MEDIA_DIR, 'image_segmentation'), exist_ok=True)
    # Add more as needed

ensure_dirs()
