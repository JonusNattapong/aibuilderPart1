import os
from transformers import AutoTokenizer, AutoModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
# Determine the base path (assuming this script is in the 'Script' folder)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the target directory to save the model
SAVE_DIRECTORY = os.path.join(BASE_PATH, 'Model', MODEL_NAME.split('/')[-1]) # Create a subfolder named after the model

# --- Main Download Logic ---
def download_model():
    """Downloads and saves the specified Hugging Face model and tokenizer."""
    logging.info(f"Attempting to download model: {MODEL_NAME}")
    logging.info(f"Target save directory: {SAVE_DIRECTORY}")

    # Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIRECTORY):
        try:
            os.makedirs(SAVE_DIRECTORY)
            logging.info(f"Created directory: {SAVE_DIRECTORY}")
        except OSError as e:
            logging.error(f"Error creating directory {SAVE_DIRECTORY}: {e}")
            return

    try:
        # Download and load tokenizer
        logging.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Save tokenizer
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        logging.info(f"Tokenizer saved successfully to {SAVE_DIRECTORY}")

        # Download and load model
        logging.info("Downloading model... (This may take a while)")
        # Use AutoModel for a general base model, or specify the head if needed
        # e.g., AutoModelForSequenceClassification, AutoModelForMaskedLM
        model = AutoModel.from_pretrained(MODEL_NAME)
        # Save model
        model.save_pretrained(SAVE_DIRECTORY)
        logging.info(f"Model saved successfully to {SAVE_DIRECTORY}")

        logging.info("Model download and saving complete.")

    except Exception as e:
        logging.error(f"An error occurred during download or saving: {e}")
        logging.error("Please check the model name and your internet connection.")

if __name__ == "__main__":
    download_model()
