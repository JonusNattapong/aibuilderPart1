"""
Utility functions for NLP dataset generation.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForCausalLM,
    pipeline
)
from config import (
    TASK_CONFIG, MODEL_CONFIG, MAX_TEXT_LENGTH,
    SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
)

class NLPTaskManager:
    """Manages NLP tasks and their processing."""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.tokenizer = None
        self.max_length = MAX_TEXT_LENGTH

    def process_batch(self, model: torch.nn.Module, texts: List[str], config: Dict[str, Any]) -> List[Dict]:
        """Process a batch of texts for the specified task."""
        
        if self.task_name == "text_classification":
            return self._process_classification(model, texts, config)
        elif self.task_name == "token_classification":
            return self._process_token_classification(model, texts, config)
        elif self.task_name == "text_generation":
            return self._process_generation(model, texts, config)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

    def _process_classification(self, model, texts, config):
        """Process texts for classification."""
        results = []
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer or model.config.tokenizer_class,
            device=MODEL_CONFIG["device"]
        )
        
        outputs = classifier(texts, batch_size=config.get("batch_size", 32))
        
        for output in outputs:
            # Get top K predictions
            top_k = min(config.get("top_k", 5), len(output))
            labels = sorted(
                [{"label": item["label"], "confidence": item["score"]}
                 for item in output[:top_k]],
                key=lambda x: x["confidence"],
                reverse=True
            )
            results.append({"labels": labels})
        
        return results

    def _process_token_classification(self, model, texts, config):
        """Process texts for token classification (NER, POS, etc.)."""
        results = []
        ner = pipeline(
            "token-classification",
            model=model,
            tokenizer=self.tokenizer or model.config.tokenizer_class,
            device=MODEL_CONFIG["device"],
            aggregation_strategy="simple"
        )
        
        outputs = ner(texts)
        
        for text_entities in outputs:
            # Filter by confidence threshold
            conf_threshold = config.get("confidence_threshold", 0.5)
            filtered_entities = [
                {
                    "entity": e["entity_group"],
                    "text": e["word"],
                    "start": e["start"],
                    "end": e["end"],
                    "confidence": e["score"]
                }
                for e in text_entities
                if e["score"] >= conf_threshold
            ]
            results.append({"entities": filtered_entities})
        
        return results

    def _process_generation(self, model, texts, config):
        """Process texts for text generation."""
        results = []
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer or model.config.tokenizer_class,
            device=MODEL_CONFIG["device"]
        )
        
        max_new_tokens = config.get("max_new_tokens", 50)
        temperature = config.get("temperature", 0.7)
        
        outputs = generator(
            texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=1
        )
        
        for output in outputs:
            generated_text = output[0]["generated_text"]
            # Remove input prompt if specified
            if not config.get("include_prompt", True):
                generated_text = generated_text.replace(texts[0], "").strip()
            
            results.append({"generated_text": generated_text})
        
        return results

def get_supported_tasks() -> Dict[str, Dict]:
    """Get dictionary of supported NLP tasks."""
    return TASK_CONFIG

def get_supported_models(task: str) -> List[str]:
    """Get list of supported models for a task."""
    return TASK_CONFIG[task].get("models", ["huggingface/default-model"])

def load_model(task: str, model_name: str) -> torch.nn.Module:
    """Load the specified model for a task."""
    try:
        if task == "text_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        elif task == "text_generation":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        model.to(MODEL_CONFIG["device"])
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name} for task {task}: {str(e)}")

def process_text(text: str, task: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Pre-process text according to task requirements."""
    # Basic cleaning
    text = text.strip()
    
    # Truncate if needed
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    # Task-specific processing
    if task == "text_classification":
        # Remove multiple spaces and newlines
        text = " ".join(text.split())
    elif task == "token_classification":
        # Preserve spacing for token positions
        pass
    elif task == "text_generation":
        # Ensure text ends with space for generation
        if not text.endswith(" "):
            text += " "
    
    return text

def save_dataset(data: List[Dict],
                filename: str,
                format_type: str,
                output_dir: str) -> str:
    """Save dataset in specified format."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_filename}.{format_type.lower()}")
    
    try:
        if format_type.upper() == "CSV":
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        elif format_type.upper() == "JSON":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type.upper() == "JSONL":
            with open(output_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error saving dataset: {str(e)}")

def setup_device() -> torch.device:
    """Set up compute device (CPU/GPU)."""
    if torch.cuda.is_available() and MODEL_CONFIG["device"] == "cuda":
        return torch.device('cuda')
    else:
        return torch.device('cpu')