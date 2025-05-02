import os
import pandas as pd
import argparse
import uuid
from config_audio import OUTPUT_DIR, BASE_PATH # Use config_audio for consistency

def find_audio_files(input_dir, recursive=True):
    """Finds audio files in the specified directory."""
    # Common audio extensions - add more if needed
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    audio_paths = []
    labels = []

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return [], []

    print(f"Scanning directory: {input_dir} (Recursive: {recursive})")

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(audio_extensions):
                full_path = os.path.join(root, file)
                # Use relative path from the base path for portability
                relative_path = os.path.relpath(full_path, BASE_PATH).replace('\\', '/')
                audio_paths.append(relative_path)

                # Infer label from the immediate parent directory name
                parent_dir_name = os.path.basename(root)
                # Avoid using the root input directory name as a label if not recursive or at top level
                if parent_dir_name != os.path.basename(input_dir.rstrip('/\\')):
                    labels.append(parent_dir_name)
                else:
                    # Label as 'unknown' if file is directly in input_dir
                    labels.append("unknown")

        if not recursive:
            # Stop searching subdirectories if not recursive
            break

    print(f"Found {len(audio_paths)} audio files.")
    return audio_paths, labels

def create_dataset_csv(audio_paths, labels, output_filename):
    """Creates a CSV file from the list of audio paths and labels."""
    if not audio_paths:
        print("No audio files found to create a dataset.")
        return

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create DataFrame
    data = {'id': [str(uuid.uuid4()) for _ in audio_paths],
            'audio_path': audio_paths}
    # Add label column only if there are meaningful labels (more than one unique label, excluding 'unknown' if it's the only one)
    meaningful_labels = [lbl for lbl in labels if lbl != "unknown"]
    if labels and len(labels) == len(audio_paths) and len(set(meaningful_labels)) > 0:
        data['label'] = labels
        header = ['id', 'audio_path', 'label']
    else:
        print("No meaningful directory-based labels found (or only 'unknown'). CSV will contain 'id' and 'audio_path'.")
        header = ['id', 'audio_path']


    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8', columns=header)
    print(f"Successfully created dataset CSV: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a CSV dataset from audio files in a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(BASE_PATH, 'placeholder_audio'), # Default to placeholder_audio
        help="Directory containing the audio files. Defaults to 'placeholder_audio' relative to the project root."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="custom_audio_dataset.csv",
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
    parser.set_defaults(recursive=True) # Default behavior is recursive

    args = parser.parse_args()

    # Make input_dir absolute or relative to BASE_PATH if not absolute
    if not os.path.isabs(args.input_dir):
        input_dir_abs = os.path.join(BASE_PATH, args.input_dir)
    else:
        input_dir_abs = args.input_dir

    audio_paths, dir_labels = find_audio_files(input_dir_abs, args.recursive)

    # Create CSV with the found paths and labels
    create_dataset_csv(audio_paths, dir_labels, args.output_filename)

    print("\nAudio dataset creation process finished.")
