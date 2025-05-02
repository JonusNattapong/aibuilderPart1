# -*- coding: utf-8 -*-
"""
train_text_classification_model.py

This script trains a Text Classification model using the Hugging Face
Transformers library on a generated Thai text classification dataset.
"""

# Import libraries
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os
import logging
import numpy as np
from datetime import datetime
import json
from benchmark_utils import compute_extended_metrics, measure_training_performance, benchmark_inference, save_benchmark_results

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# !! IMPORTANT: Adjust DATASET_PATH to your actual classification dataset CSV !!
DATASET_PATH = os.path.join(BASE_PATH, 'DataOutput/thai_dataset_text_classification.csv') # Example path
MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, 'Models/text_classification_model')
LOGGING_DIR = os.path.join(BASE_PATH, 'Script/logs_classification')

# Model checkpoint for tokenizer and base model
MODEL_CHECKPOINT = "airesearch/wangchanberta-base-att-spm-uncased" # Can use the same base model

# Tokenization parameters
MAX_LENGTH = 256 # Adjust as needed for classification tasks

# Training parameters
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1" # Use f1 or accuracy for classification
FP16 = torch.cuda.is_available()

# --- End Configuration ---

def load_and_prepare_dataset(dataset_path):
    """Loads the classification dataset from CSV and prepares it."""
    logging.info(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        logging.info("Dataset loaded successfully.")
        logging.info(f"Initial dataset shape: {df.shape}")

        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        # Handle potential missing values
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int) # Ensure labels are integers

        logging.info(f"Dataset shape after cleaning: {df.shape}")
        if len(df) == 0:
            raise ValueError("No valid examples found in the dataset after cleaning.")

        # Convert pandas DataFrame to Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(df)

        # Create ClassLabel feature (important for classification)
        # This assumes labels are 0, 1, ... num_classes-1
        num_classes = df['label'].nunique()
        class_names = [str(i) for i in range(num_classes)] # Example names
        hf_dataset = hf_dataset.cast_column("label", ClassLabel(num_classes=num_classes, names=class_names))


        # Split into train/validation sets (e.g., 90% train, 10% validation)
        train_test_split = hf_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })

        logging.info("Dataset prepared for Hugging Face:")
        logging.info(dataset_dict)
        logging.info(f"Number of classes: {num_classes}")
        return dataset_dict, num_classes

    except FileNotFoundError:
        logging.error(f"Error: Dataset file not found at {dataset_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during dataset loading/preparation: {e}")
        raise

def tokenize_function(examples, tokenizer, max_length):
    """Tokenizes the text data."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

# Use compute_metrics from benchmark_utils
compute_metrics = compute_extended_metrics

def main():
    """Main function to run the training pipeline."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    logging.info(f"Model output directory: {MODEL_OUTPUT_DIR}")
    logging.info(f"Logging directory: {LOGGING_DIR}")

    # 1. Load and Prepare Dataset
    dataset_dict, num_classes = load_and_prepare_dataset(DATASET_PATH)

    # 2. Tokenization
    logging.info(f"Loading tokenizer from: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    logging.info("Tokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH}
    )
    # Remove original text column after tokenization
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # Rename label column if necessary (Trainer expects 'labels')
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Set format for PyTorch
    tokenized_datasets.set_format("torch")

    logging.info("Tokenization complete.")
    logging.info(tokenized_datasets)

    # 3. Load Model
    logging.info(f"Loading pre-trained model for Sequence Classification from: {MODEL_CHECKPOINT}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_classes)

    # 4. Define Training Arguments
    logging.info("Defining training arguments...")
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        fp16=FP16,
        report_to="tensorboard",
        seed=42,
        save_total_limit=2, # Optional: Limit number of checkpoints saved
    )

    # 5. Initialize Trainer
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics # Add metrics computation
    )

    # 6. Start Training
    logging.info("Starting training...")
    try:
        # Use measure_training_performance from benchmark_utils
        train_result, performance_metrics = measure_training_performance(
            trainer,
            lambda: trainer.train()
        )
        
        # Update training metrics with performance metrics
        train_result.metrics.update(performance_metrics)
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise

    # 7. Evaluate Model
    logging.info("Evaluating model...")
    try:
        eval_metrics = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_metrics}")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")

    # 8. Save Model
    logging.info(f"Saving the best model to {MODEL_OUTPUT_DIR}...")
    try:
        trainer.save_model(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        logging.info("Model and tokenizer saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred during model saving: {e}")
        raise

    # Run benchmarking
    logging.info("\nRunning benchmarking tests...")
    test_texts = [
        "ข้อความนี้เกี่ยวกับการเมือง",
        "ฉันรู้สึกมีความสุขมากวันนี้",
        "อยากไปเที่ยวต่างประเทศ",
        "วิธีการเขียนโค้ดที่ดี"
    ]
    
    benchmark_results = benchmark_inference(MODEL_OUTPUT_DIR, test_texts)
    
    # Save all metrics using benchmark utils
    model_config = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "batch_size": TRAIN_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS
    }
    
    save_benchmark_results(
        MODEL_OUTPUT_DIR,
        train_result.metrics,
        eval_metrics,
        benchmark_results,
        model_config
    )
    
    logging.info("\n--- Training, Evaluation and Benchmarking Complete ---")
    logging.info(f"Model saved in: {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

    # Optional: Add code here to test the trained model after training
    # from transformers import pipeline
    # try:
    #     logging.info("Testing the trained classification model...")
    #     classifier = pipeline(
    #         "text-classification",
    #         model=MODEL_OUTPUT_DIR,
    #         tokenizer=MODEL_OUTPUT_DIR,
    #         device=0 if torch.cuda.is_available() else -1
    #     )
    #     test_text = "ข้อความนี้เกี่ยวกับการเมือง" # Replace with your test text
    #     result = classifier(test_text)
    #     logging.info(f"Test Text: {test_text}")
    #     logging.info(f"Test Result: {result}")
    # except Exception as e:
    #     logging.error(f"Failed to test the pipeline after training: {e}")

