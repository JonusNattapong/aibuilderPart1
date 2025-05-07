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
    for category in MODELS.keys():
        os.makedirs(os.path.join(BASE_MODEL_DIR, category), exist_ok=True)

def get_model_processor(category, model_path, cache_dir):
    """Get appropriate processor based on model category"""
    if category in ['text', 'code', 'translation']:
        return AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    elif category in ['vision', 'creative']:
        return AutoFeatureExtractor.from_pretrained(model_path, cache_dir=cache_dir)
    elif category in ['audio', 'speech', 'multimodal']:
        return AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    else:
        return None

def download_model(model_name, model_path, category):
    """Download a specific model"""
    try:
        logger.info(f"Downloading {model_name} model...")
        cache_dir = os.path.join(BASE_MODEL_DIR, category, model_name)
        
        # Download model files
        snapshot_download(
            repo_id=model_path,
            cache_dir=cache_dir,
            local_dir=cache_dir
        )
        
        # Load and save model components
        try:
            processor = get_model_processor(category, model_path, cache_dir)
            model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
            
            if processor:
                processor.save_pretrained(cache_dir)
            model.save_pretrained(cache_dir)
            
        except Exception as e:
            logger.warning(f"Could not load model components: {str(e)}")
            logger.info("Continuing with raw file download only")
        
        logger.info(f"Successfully downloaded {model_name} model to {cache_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {model_name} model: {str(e)}")
        return False

def load_model(category, model_name):
    """Load a specific model from the local directory"""
    model_path = os.path.join(BASE_MODEL_DIR, category, model_name)
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    try:
        components = {}
        
        # Load processor/tokenizer if available
        try:
            processor = get_model_processor(category, model_path, model_path)
            if processor:
                components['processor'] = processor
        except Exception as e:
            logger.warning(f"Could not load processor: {str(e)}")
        
        # Load model
        try:
            model = AutoModel.from_pretrained(model_path)
            components['model'] = model
        except Exception as e:
            logger.warning(f"Could not load model: {str(e)}")
        
        return components if components else None
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None

def download_category_models(category):
    """Download all models for a specific category"""
    if category not in MODELS:
        logger.error(f"Unknown category: {category}")
        return False
    
    success = True
    for model_name, model_path in MODELS[category].items():
        if not download_model(model_name, model_path, category):
            success = False
    return success

def download_all_models():
    """Download all models for all categories"""
    create_model_dir()
    
    total_models = sum(len(models) for models in MODELS.values())
    successful = 0
    
    for category in MODELS.keys():
        logger.info(f"\nDownloading {category} models...")
        for model_name, model_path in MODELS[category].items():
            if download_model(model_name, model_path, category):
                successful += 1
    
    logger.info(f"\nDownload complete: {successful}/{total_models} models downloaded successfully")
    return successful == total_models

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download AI models for different data types')
    parser.add_argument('--category', choices=list(MODELS.keys()) + ['all'],
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
                logger.info(f"Successfully loaded {args.model_name} from {os.path.join(BASE_MODEL_DIR, category, args.model_name)}")
        else:
            logger.error(f"Model {args.model_name} not found in any category")
    else:
        if args.category == 'all' or not args.category:
            success = download_all_models()
        else:
            success = download_category_models(args.category)
        
        exit(0 if success else 1)