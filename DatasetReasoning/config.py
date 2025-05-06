"""
Configuration settings for reasoning dataset generation.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_CACHE_DIR = BASE_DIR / ".model_cache"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Text settings
MAX_PROMPT_LENGTH = 1024
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8

# Supported reasoning tasks
TASK_CONFIG = {
    "cot": {
        "name": "Chain of Thought",
        "description": "Step-by-step reasoning approach",
        "models": [
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "EleutherAI/gpt-neo-2.7B"
        ],
        "parameters": {
            "max_tokens": {
                "name": "Maximum Tokens",
                "type": "number",
                "min": 100,
                "max": 1000,
                "default": 512
            },
            "temperature": {
                "name": "Temperature",
                "type": "number",
                "min": 0.1,
                "max": 2.0,
                "default": 0.7
            }
        }
    },
    "react": {
        "name": "ReAct (Reasoning + Acting)",
        "description": "Reasoning with action steps",
        "models": [
            "google/flan-t5-xl",
            "EleutherAI/gpt-neo-2.7B",
            "facebook/opt-6.7b"
        ],
        "parameters": {
            "max_tokens": {
                "name": "Maximum Tokens",
                "type": "number",
                "min": 100,
                "max": 1000,
                "default": 512
            },
            "temperature": {
                "name": "Temperature",
                "type": "number",
                "min": 0.1,
                "max": 2.0,
                "default": 0.7
            }
        }
    },
    "tot": {
        "name": "Tree of Thought",
        "description": "Explore multiple reasoning paths",
        "models": [
            "EleutherAI/gpt-neo-2.7B",
            "facebook/opt-6.7b",
            "google/flan-t5-xl"
        ],
        "parameters": {
            "max_tokens": {
                "name": "Maximum Tokens",
                "type": "number",
                "min": 100,
                "max": 1000,
                "default": 512
            },
            "temperature": {
                "name": "Temperature",
                "type": "number",
                "min": 0.1,
                "max": 2.0,
                "default": 0.8
            },
            "num_branches": {
                "name": "Number of Branches",
                "type": "number",
                "min": 2,
                "max": 5,
                "default": 3
            }
        }
    },
    "meta": {
        "name": "Meta Reasoning",
        "description": "Strategy selection and monitoring",
        "models": [
            "EleutherAI/gpt-neo-2.7B",
            "google/flan-t5-xl",
            "facebook/opt-6.7b"
        ],
        "parameters": {
            "max_tokens": {
                "name": "Maximum Tokens",
                "type": "number",
                "min": 100,
                "max": 1000,
                "default": 512
            },
            "temperature": {
                "name": "Temperature",
                "type": "number",
                "min": 0.1,
                "max": 2.0,
                "default": 0.7
            }
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    "use_fp16": os.environ.get("USE_FP16", "1") == "1",
    "device": "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu",
    "num_workers": int(os.environ.get("NUM_WORKERS", "4")),
    "pin_memory": True,
    "cache_dir": MODEL_CACHE_DIR
}

# Cache settings
ENABLE_CACHE = True
CACHE_DIR = BASE_DIR / ".cache"
CACHE_TTL = 24 * 60 * 60  # 24 hours
MAX_CACHE_SIZE = 1000  # entries

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_FILENAME = "reasoning_dataset.log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Create additional directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Templates for different reasoning approaches
TEMPLATES = {
    "cot": {
        "prompt": "Let's solve this step by step:\nQuestion: {prompt}\nLet's approach this step by step:\n1.",
        "step_prefix": "\n{step_num}.",
        "answer_prefix": "\nTherefore, "
    },
    "react": {
        "prompt": "Let's approach this by Reasoning and Acting:\nQuestion: {prompt}\nThought 1: Let me break this down.",
        "thought_prefix": "\nThought {step_num}:",
        "action_prefix": "\nAction {step_num}:",
        "observation_prefix": "\nObservation {step_num}:",
        "answer_prefix": "\nFinal Answer:"
    },
    "tot": {
        "prompt": "Let's solve this using Tree of Thought reasoning:\nProblem: {prompt}\nLet's explore different approaches:\nBranch 1:",
        "branch_prefix": "\nBranch {branch_num}:",
        "step_prefix": "\nStep {step_num}:",
        "evaluation_prefix": "\nEvaluation:",
        "confidence_prefix": "\nConfidence:"
    },
    "meta": {
        "prompt": "Let's solve this using meta-reasoning:\nProblem: {prompt}\n\n1. Strategy Selection:",
        "sections": [
            "1. Strategy Selection:",
            "2. Reasoning Process:",
            "3. Monitoring:",
            "4. Evaluation:",
            "Final Answer:"
        ]
    }
}