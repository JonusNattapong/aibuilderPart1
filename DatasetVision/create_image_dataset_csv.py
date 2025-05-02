import os
import pandas as pd
import argparse
import uuid
from config_vision import OUTPUT_DIR, BASE_PATH
# Added imports for local model inference
import torch
from PIL import Image
try:
    from transformers import AutoProcessor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: `transformers` library not found. Local model classification will not be available.")
    print("Install it with: pip install transformers torch")


def find_image_files(input_dir, recursive=True):
    """Finds image files in the specified directory."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_paths = []
    labels = []

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return [], []

    print(f"Scanning directory: {input_dir} (Recursive: {recursive})")

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                # Use relative path from the base path for portability
                relative_path = os.path.relpath(full_path, BASE_PATH).replace('\\', '/')
                image_paths.append(relative_path)

                # Infer label from the immediate parent directory name
                parent_dir_name = os.path.basename(root)
                # Avoid using the root input directory name as a label if not recursive or at top level
                if parent_dir_name != os.path.basename(input_dir.rstrip('/\\')):
                    labels.append(parent_dir_name)
                else:
                    labels.append("unknown") # Or None, or empty string

        if not recursive:
            # Stop searching subdirectories if not recursive
            break

    print(f"Found {len(image_paths)} image files.")
    return image_paths, labels

# New function for local label generation
def generate_local_labels(image_paths, model_name):
    """Generates labels for images using a local Hugging Face model."""
    if not TRANSFORMERS_AVAILABLE:
        print("Error: Transformers library is required for local model classification.")
        return None # Indicate failure

    print(f"\nLoading local classification model: {model_name}...")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None # Indicate failure

    generated_labels = []
    total_images = len(image_paths)
    print(f"Generating labels for {total_images} images...")

    with torch.no_grad():
        for i, relative_path in enumerate(image_paths):
            full_path = os.path.join(BASE_PATH, relative_path)
            try:
                image = Image.open(full_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)

                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                label = model.config.id2label[predicted_class_idx]
                generated_labels.append(label)
                print(f"  Processed {i+1}/{total_images}: {relative_path} -> {label}")

            except FileNotFoundError:
                print(f"Error: Image file not found at {full_path}. Skipping.")
                generated_labels.append("error_not_found")
            except Exception as e:
                print(f"Error processing image {full_path}: {e}")
                generated_labels.append("error_processing")

    print("Finished generating labels locally.")
    return generated_labels


def create_dataset_csv(image_paths, labels, output_filename):
    """Creates a CSV file from the list of image paths and labels."""
    if not image_paths:
        print("No image files found to create a dataset.")
        return

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create DataFrame
    data = {'id': [str(uuid.uuid4()) for _ in image_paths],
            'image_path': image_paths}
    if labels and len(labels) == len(image_paths) and len(set(labels)) > 1: # Add label column if meaningful labels exist
        data['label'] = labels
        header = ['id', 'image_path', 'label']
    else:
        header = ['id', 'image_path']

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8', columns=header)
    print(f"Successfully created dataset CSV: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a CSV dataset from image files in a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        # required=True, # No longer required as it has a default
        default=os.path.join(BASE_PATH, 'DatasetVision', 'Img'), # Set default input directory
        help="Directory containing the image files. Defaults to 'DatasetVision/Img' relative to the project root."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="custom_image_dataset.csv",
        help="Name of the output CSV file (will be saved in DataOutput)."
    )
    parser.add_argument(
        "--recursive",
        action='store_true', # Default is False if not specified
        help="Scan subdirectories recursively."
    )
    parser.add_argument(
        "--no-recursive",
        action='store_false',
        dest='recursive',
        help="Do not scan subdirectories."
    )
    # New argument for local model
    parser.add_argument(
        "--local_classifier_model",
        type=str,
        default=None,
        help="Optional: Name or path of a local Hugging Face image classification model to generate labels. Requires `transformers` and `torch`. Overrides directory-based labels."
    )
    parser.set_defaults(recursive=True) # Default behavior is recursive

    args = parser.parse_args()

    # Make input_dir absolute or relative to BASE_PATH if not absolute
    # If the default is used, it's already absolute based on BASE_PATH
    if not os.path.isabs(args.input_dir):
        input_dir_abs = os.path.join(BASE_PATH, args.input_dir)
    else:
        input_dir_abs = args.input_dir

    image_paths, dir_labels = find_image_files(input_dir_abs, args.recursive)

    final_labels = dir_labels # Start with directory labels

    # Generate labels locally if model specified
    if args.local_classifier_model:
        if not TRANSFORMERS_AVAILABLE:
             print("Cannot generate local labels because `transformers` is not installed.")
        else:
            print(f"Attempting to generate labels using local model: {args.local_classifier_model}")
            model_labels = generate_local_labels(image_paths, args.local_classifier_model)
            if model_labels is not None:
                final_labels = model_labels # Override with model labels if successful
            else:
                print("Failed to generate labels using the local model. Falling back to directory-based labels (if any).")

    # Create CSV with the final set of labels
    create_dataset_csv(image_paths, final_labels, args.output_filename)

    print("\nDataset creation process finished.")
