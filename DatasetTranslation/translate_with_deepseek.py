import os
import sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Install required packages if not already installed
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])

# Model and device setup
MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/deepseek-llm-7b-chat")
DEVICE = 0 if torch.cuda.is_available() else -1

def get_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        device=DEVICE
    )

def build_prompt(text):
    return (
        "You are a professional medical translator. "
        "Translate the following English medical text to Thai with high accuracy, preserving all medical terminology and context.\n\n"
        f"English:\n{text}\n\nThai:"
    )

def translate_medical(text, pipe=None):
    if pipe is None:
        pipe = get_pipeline()
    prompt = build_prompt(text)
    output = pipe(prompt)
    # Extract only the Thai translation after "Thai:"
    result = output[0]['generated_text']
    if "Thai:" in result:
        return result.split("Thai:")[-1].strip()
    return result.strip()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical translation (English to Thai) using DeepSeek LLM.")
    parser.add_argument("--text", type=str, required=True, help="English medical text to translate")
    args = parser.parse_args()
    pipe = get_pipeline()
    translation = translate_medical(args.text, pipe)
    print("=== Translation Result ===")
    print(translation)