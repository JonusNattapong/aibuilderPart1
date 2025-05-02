# -*- coding: utf-8 -*-
"""
Script to download datasets from Hugging Face Hub and save them locally.

Example Usage:
python Script/download_hf_datasets.py --datasets USERNAME/DATASET_NAME1 USERNAME/DATASET_NAME2
python Script/download_hf_datasets.py -d USERNAME/DATASET_NAME

Requires the 'datasets' library: pip install datasets
"""

import os
import logging
import argparse
from datasets import load_dataset, DownloadConfig
from huggingface_hub import HfFolder # To potentially save token

# --- Configuration ---
# Determine the base path (assuming this script is in the 'Script' folder)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the target base directory to save datasets
DEFAULT_DOWNLOAD_DIR = os.path.join(BASE_PATH, 'DatasetDownload')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face Hub.")
    parser.add_argument(
        "-d", "--datasets",
        nargs='+',  # Allows specifying one or more dataset IDs
        required=True,
        help="List of dataset IDs from Hugging Face Hub (e.g., 'username/dataset_name' or 'dataset_name')."
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"The base directory where datasets will be saved. Defaults to: {DEFAULT_DOWNLOAD_DIR}"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face Hub token for accessing private datasets. If not provided, uses cached login or public access."
    )
    parser.add_argument(
        "--save_token",
        action="store_true",
        help="If provided with --token, save the token for future use."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional: Specify a specific subset/configuration of the dataset to download."
    )
    return parser.parse_args()

# --- Download Logic ---
def download_and_save_datasets(dataset_ids, base_download_dir, token=None, subset=None):
    """
    Downloads datasets from Hugging Face Hub and saves them to disk.

    Args:
        dataset_ids (list): A list of dataset IDs to download.
        base_download_dir (str): The base directory to save datasets.
        token (str, optional): Hugging Face Hub token. Defaults to None.
        subset (str, optional): Specific dataset subset/configuration. Defaults to None.
    """
    logging.info(f"Base download directory set to: {base_download_dir}")
    if not os.path.exists(base_download_dir):
        try:
            os.makedirs(base_download_dir)
            logging.info(f"Created base directory: {base_download_dir}")
        except OSError as e:
            logging.error(f"Error creating base directory {base_download_dir}: {e}")
            return # Stop if base directory cannot be created

    # Configure download settings, including the token if provided
    download_config = DownloadConfig(use_auth_token=token)

    for dataset_id in dataset_ids:
        logging.info(f"--- Processing dataset: {dataset_id} ---")
        # Create a safe directory name from the dataset ID
        safe_subdir_name = dataset_id.replace('/', '_') # Replace slashes for directory compatibility
        target_dir = os.path.join(base_download_dir, safe_subdir_name)

        logging.info(f"Target save directory: {target_dir}")

        if os.path.exists(target_dir):
            logging.warning(f"Directory already exists: {target_dir}. Skipping download for {dataset_id}. Delete the directory to re-download.")
            continue

        try:
            # Load the dataset (this handles the download)
            logging.info(f"Attempting to download '{dataset_id}'{f' (subset: {subset})' if subset else ''}...")
            dataset = load_dataset(
                dataset_id,
                name=subset, # Pass subset name if provided
                download_config=download_config
            )
            logging.info(f"Successfully loaded dataset '{dataset_id}'.")

            # Save the dataset to the target directory
            logging.info(f"Saving dataset to {target_dir}...")
            dataset.save_to_disk(target_dir)
            logging.info(f"Successfully saved '{dataset_id}' to {target_dir}")

        except Exception as e:
            logging.error(f"Failed to download or save dataset '{dataset_id}': {e}")
            # Clean up potentially incomplete directory
            if os.path.exists(target_dir):
                 try:
                     # Be cautious with recursive deletion, maybe just log the error
                     logging.warning(f"An error occurred. The directory {target_dir} might contain incomplete data.")
                     # import shutil
                     # shutil.rmtree(target_dir)
                     # logging.info(f"Removed potentially incomplete directory: {target_dir}")
                 except Exception as cleanup_e:
                     logging.error(f"Error during cleanup of {target_dir}: {cleanup_e}")
        logging.info(f"--- Finished processing dataset: {dataset_id} ---")


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # Handle token saving
    if args.token and args.save_token:
        HfFolder.save_token(args.token)
        logging.info("Hugging Face token saved.")
        # Use the provided token for this run even if saving
        token_to_use = args.token
    elif args.token:
        # Use the provided token without saving it globally
        token_to_use = args.token
    else:
        # Rely on cached token or public access
        token_to_use = None

    download_and_save_datasets(
        dataset_ids=args.datasets,
        base_download_dir=args.download_dir,
        token=token_to_use,
        subset=args.subset
    )
    logging.info("Dataset download process finished.")
