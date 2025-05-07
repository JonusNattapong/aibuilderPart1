#!/bin/bash

# Create cache directory
mkdir -p ~/.cache/huggingface/hub

# Download model files
model_name="microsoft/mdeberta-v3-base"
files=(
    "config.json"
    "pytorch_model.bin"
    "special_tokens_map.json" 
    "tokenizer_config.json"
    "tokenizer.json"
    "vocab.json"
)

echo "Downloading ${model_name} files..."
for file in "${files[@]}"; do
    wget "https://huggingface.co/${model_name}/resolve/main/${file}" -P ~/.cache/huggingface/hub/${model_name}
done

echo "Model files downloaded to: ~/.cache/huggingface/hub/${model_name}"