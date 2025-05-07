# Specialized AI Models Setup

This directory contains scripts for downloading and setting up specialized AI models across multiple domains.

## Model Categories and Available Models

### Vision Models
- **captioning** (llava-1.5-7b-hf) - Image captioning
- **stable-diffusion** - Image generation from text
- **clip** - Image and text matching
- **pix2pix** - Image-to-image translation
- **cycleGAN** - Unpaired image translation
- **YOLO** - Object detection
- **Faster-RCNN** - Object detection with bounding boxes
- **Mask-RCNN** - Instance segmentation
- **U-Net** - Medical image segmentation

### Text Models
- **qna** (Mistral-7B) - Question answering
- **summarization** (BART) - Text summarization
- **text-gen** (GPT-2) - Text generation
- **translation** (mBART) - Language translation
- **semantic-search** (MiniLM) - Semantic search
- **chatbot** (DialoGPT) - Conversational AI
- **RoBERTa** - Text classification
- **DistilBERT** - Lightweight BERT
- **XLNet** - Advanced language understanding

### Audio Models
- **bg-music** (MusicGen) - Music generation
- **speech-to-text** (Whisper) - Speech recognition
- **text-to-speech** (Tacotron2) - Voice synthesis
- **sound-effect** - Sound effect generation
- **DeepNoise** - Noise reduction
- **SEGAN** - Speech enhancement

### Speech Models
- **whisper** - Speech recognition
- **wav2vec2** - Speech processing
- **hubert** - Speech understanding
- **text-to-speech** (FastSpeech2) - Voice synthesis

### Code Generation Models
- **html-gen** (DeepSeek) - HTML/CSS generation
- **python-gen** (CodeGen) - Python code generation
- **javascript-gen** (DeepSeek) - JavaScript generation
- **sql-gen** (DeepSeek) - SQL query generation

### Creative Models
- **chroma** - Artistic content generation
- **stable-diffusion** - Image generation

### Multimodal Models
- **clip** - Vision-language understanding
- **vilt** - Visual-language tasks

### Additional Categories
- **Time Series Models**: prophet, lstm
- **Recommendation Models**: user-item-cf, content-based
- **Translation Models**: mBART, T5, MarianMT

## Installation

Models are stored in `D:\Models` with category-based organization:
```
D:\Models\
├── vision\
├── text\
├── audio\
├── speech\
├── code\
├── creative\
├── multimodal\
├── time-series\
├── recommendation\
└── translation\
```

### Usage

Download all models:
```bash
python download_specialized_models.py --category all
```

Download specific category:
```bash
python download_specialized_models.py --category vision
python download_specialized_models.py --category text
# etc...
```

Load models in code:
```python
from download_specialized_models import load_model

# Load a vision model
vision_model = load_model('vision', 'YOLO')
if vision_model:
    model = vision_model['model']
    processor = vision_model.get('processor')

# Load a text model
text_model = load_model('text', 'qna')
if text_model:
    model = text_model['model']
    processor = text_model.get('processor')
```

## Storage Requirements

Estimated storage requirements by category:
- Vision Models: ~20GB
- Text Models: ~15GB
- Audio Models: ~10GB
- Speech Models: ~8GB
- Code Models: ~12GB
- Creative Models: ~7GB
- Multimodal Models: ~5GB
- Other Categories: ~5GB each

Total storage requirement: ~90GB recommended

## System Requirements

- Python 3.8 or later
- CUDA-capable GPU recommended for faster inference
- Minimum 16GB RAM (32GB recommended)
- At least 100GB free space on D: drive
- Internet connection for downloading models

## Troubleshooting

1. Memory Issues
   ```bash
   # Set maximum GPU memory usage
   export CUDA_VISIBLE_DEVICES="0"  # Use specific GPU
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
   ```

2. Download Issues
   - Check internet connection
   - Verify D: drive accessibility
   - Try downloading models individually

3. Loading Issues
   ```python
   # Load model in CPU mode if GPU memory is insufficient
   model = load_model('category', 'model_name', device='cpu')
   ```

## Notes

- Models are downloaded from Hugging Face Hub
- Some models may require additional dependencies
- Large models may take significant time to download
- Models are cached to avoid redownloading
- Check model licenses before commercial use