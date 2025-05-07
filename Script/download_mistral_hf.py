import os
from huggingface_hub import snapshot_download

def download_model(model_id="mistralai/Mistral-7B-Instruct-v0.3", output_dir="models"):
    model_path = os.path.join(output_dir, model_id.split('/')[-1])
    os.makedirs(model_path, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"Model downloaded to: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download Mistral model from Hugging Face')
    parser.add_argument('--model_id', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help='Model ID from Hugging Face')
    parser.add_argument('--output_dir', type=str, default="models", help='Output directory for model files')
    args = parser.parse_args()
    download_model(args.model_id, args.output_dir)