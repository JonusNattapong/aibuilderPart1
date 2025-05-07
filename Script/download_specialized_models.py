import os
import torch
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from huggingface_hub import snapshot_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for models
BASE_MODEL_DIR = "D:\\Models"

# Comprehensive model dictionary
MODELS = {
    # Vision models
    'vision': {
        'captioning': 'llava-hf/llava-1.5-7b-hf',
        'stable-diffusion': 'CompVis/stable-diffusion-v-1-4-original',
        'clip': 'openai/clip-vit-base-patch32',
        'pix2pix': 'junyanz/pytorch-CycleGAN-and-pix2pix',
        'cycleGAN': 'junyanz/CycleGAN',
        'YOLO': 'ultralytics/yolov5',
        'Faster-RCNN': 'facebook/detectron2',
        'Mask-RCNN': 'facebook/maskrcnn-benchmark',
        'U-Net': 'zhixuhao/unet'
    },

    # Text models
    'text': {
        'qna': 'mistralai/Mistral-7B-Instruct-v0.2',
        'summarization': 'facebook/bart-large-cnn',
        'text-gen': 'gpt2',
        'translation': 'facebook/mbart-large-50',
        'semantic-search': 'sentence-transformers/all-MiniLM-L6-v2',
        'chatbot': 'microsoft/DialoGPT-medium',
        'RoBERTa': 'roberta-base',
        'DistilBERT': 'distilbert-base-uncased',
        'XLNet': 'xlnet-base-cased'
    },

    # Audio models
    'audio': {
        'bg-music': 'facebook/musicgen-large',
        'speech-to-text': 'openai/whisper-large',
        'text-to-speech': 'google/tacotron2',
        'sound-effect': 'facebook/audio-spectrogram-transformer',
        'DeepNoise': 'deepnoise/denoise',
        'SEGAN': 'segan/speech-enhancement'
    },

    # Speech models
    'speech': {
        'whisper': 'openai/whisper-large',
        'wav2vec2': 'facebook/wav2vec2-base',
        'hubert': 'facebook/hubert-base-ls960',
        'text-to-speech': 'facebook/fastspeech2'
    },

    # Code generation models
    'code': {
        'html-gen': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'python-gen': 'huggingface/codegen-350M-mono',
        'javascript-gen': 'deepseek-ai/deepseek-coder-javascript',
        'sql-gen': 'deepseek-ai/deepseek-coder-sql'
    },

    # Creative models
    'creative': {
        'chroma': 'lodestones/Chroma',
        'stable-diffusion': 'CompVis/stable-diffusion-v-1-4-original'
    },

    # Multimodal models
    'multimodal': {
        'clip': 'openai/clip-vit-base-patch32',
        'vilt': 'google/vilt-b32-mlm'
    },

    # Additional specialized models
    'time-series': {
        'prophet': 'facebook/prophet',
        'lstm': 'keras/lstm'
    },
    'recommendation': {
        'user-item-cf': 'google/bert-for-user-item-cf',
        'content-based': 'facebook/bert-base-uncased'
    },
    'translation': {
        'mBART': 'facebook/mbart-large-50',
        't5': 'google/t5-base',
        'marianmt': 'Helsinki-NLP/opus-mt-en-zh'
    }
}

def create_model_dir():
    """Create models directory if it doesn't exist"""
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
            return {'processor': processor, 'model': model}
            
        elif category == 'vision':
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            return {'feature_extractor': feature_extractor, 'model': model}
            
        elif category == 'multimodal':
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return {'processor': processor, 'model': model, 'tokenizer': tokenizer}
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download AI models for different data types')
    parser.add_argument('--category', choices=['audio', 'vision', 'multimodal', 'all'],
                      help='Category of models to download')
    parser.add_argument('--load', action='store_true',
                      help='Load a model after downloading')
    parser.add_argument('--model-name',
                      help='Specific model name to load (used with --load)')
    
    args = parser.parse_args()
    
    if args.load and args.model_name:
        # Find the category for the model name
        category = None
        for cat, models in MODELS.items():
            if args.model_name in models:
                category = cat
                break
        
        if category:
            model = load_model(category, args.model_name)
            if model:
                logger.info(f"Successfully loaded {args.model_name} from {get_model_path(category, args.model_name)}")
        else:
            logger.error(f"Model {args.model_name} not found in any category")
    else:
        if args.category == 'all' or not args.category:
            success = download_all_models()
        else:
            success = download_category_models(args.category)
        
        exit(0 if success else 1)