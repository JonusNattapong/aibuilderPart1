# Specialized AI Models Setup

This directory contains scripts for downloading and setting up specialized AI models for audio, vision, and multimodal processing.

## Model Storage Location

All models are stored in `D:\Models` with the following structure:
```
D:\Models\
├── audio\
│   ├── wav2vec2\
│   ├── hubert\
│   ├── whisper\
│   └── audio-spectrogram\
├── vision\
│   ├── vit\
│   ├── detr\
│   ├── deit\
│   └── segformer\
└── multimodal\
    ├── clip\
    ├── layoutlm\
    └── vilbert\
```

## Available Models

### Audio Models
- **wav2vec2**: Facebook's wav2vec2-base model for speech recognition
- **hubert**: Facebook's HuBERT model for speech processing
- **whisper**: OpenAI's Whisper model for speech recognition
- **audio-spectrogram**: Facebook's Audio Spectrogram Transformer

### Vision Models
- **vit**: Google's Vision Transformer
- **detr**: Facebook's DETR object detection model
- **deit**: Facebook's Data-efficient Image Transformer
- **segformer**: NVIDIA's SegFormer for image segmentation

### Multimodal Models
- **clip**: OpenAI's CLIP model for image-text understanding
- **layoutlm**: Microsoft's LayoutLM for document understanding
- **vilbert**: VIT-GPT2 for image captioning

## Usage

### Downloading Models

Download all models:
```bash
python download_specialized_models.py --category all
```

Download specific category:
```bash
python download_specialized_models.py --category audio
python download_specialized_models.py --category vision
python download_specialized_models.py --category multimodal
```

### Loading Models

To load a specific model:
```bash
# Download and load a model
python download_specialized_models.py --load --model-name wav2vec2

# Example usage in code:
from download_specialized_models import load_model

# Load audio model
audio_model = load_model('audio', 'wav2vec2')
if audio_model:
    processor = audio_model['processor']
    model = audio_model['model']

# Load vision model
vision_model = load_model('vision', 'vit')
if vision_model:
    feature_extractor = vision_model['feature_extractor']
    model = vision_model['model']

# Load multimodal model
multimodal_model = load_model('multimodal', 'clip')
if multimodal_model:
    processor = multimodal_model['processor']
    model = multimodal_model['model']
    tokenizer = multimodal_model['tokenizer']
```

## Requirements

See `requirements_specialized.txt` for detailed package requirements. Main dependencies include:
- PyTorch
- Transformers
- Hugging Face Hub
- Additional audio/vision processing libraries

## Troubleshooting

1. If you encounter CUDA/GPU related errors:
   - Make sure you have the correct CUDA version installed
   - Try running with CPU only by setting: `export CUDA_VISIBLE_DEVICES=""`

2. If you face download issues:
   - Check your internet connection
   - Ensure you have sufficient disk space in D: drive
   - Try downloading models individually

3. For memory issues:
   - Consider downloading models one at a time
   - Free up system memory before downloading large models

4. For storage location issues:
   - Ensure D: drive exists and is accessible
   - Check write permissions for D:\Models directory
   - If D: drive is not available, modify BASE_MODEL_DIR in the script

## Notes

- Models are downloaded from Hugging Face Hub and stored locally in D:\Models
- Downloaded models are cached to avoid redownloading
- Each model category requires different disk space:
  - Audio: ~5GB
  - Vision: ~4GB
  - Multimodal: ~7GB

## Storage Requirements

Ensure you have at least 20GB of free space on your D: drive before downloading all models. The actual space required may vary depending on the specific model versions and additional cached files.