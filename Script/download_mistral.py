import os
import argparse
from huggingface_hub import snapshot_download
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import torch

def download_model(model_id="mistralai/Mistral-7B-Instruct-v0.3", output_dir="models"):
    """Download model and tokenizer from Hugging Face"""
    print(f"\nDownloading {model_id}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_id.split('/')[-1])
    
    try:
        # Download model files
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        
        # Load and save tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)
        
        # Load and save model in safetensors format
        print("\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.save_pretrained(
            model_path,
            safe_serialization=True
        )
        
        print(f"\nModel and tokenizer saved to: {model_path}")
        
        # Verify installation
        print("\nVerifying installation...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Test tokenization and generation
        test_input = tokenizer("Hello, I am", return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**test_input, max_new_tokens=10)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print("\nTest generation successful!")
        print(f"Input: 'Hello, I am'")
        print(f"Output: '{response}'")
        
        return model_path
        
    except Exception as e:
        print(f"\nError downloading model: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download Mistral model')
    parser.add_argument('--model_id', type=str, 
                      default="mistralai/Mistral-7B-Instruct-v0.3",
                      help='Model ID from Hugging Face')
    parser.add_argument('--output_dir', type=str, 
                      default="models",
                      help='Output directory for model files')
    
    args = parser.parse_args()
    
    download_model(args.model_id, args.output_dir)

if __name__ == '__main__':
    main()