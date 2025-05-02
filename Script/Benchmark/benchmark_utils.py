# -*- coding: utf-8 -*-
"""
benchmark_utils.py

Utility functions for benchmarking model performance and resource usage.
"""

import time
import psutil
import json
import logging
import numpy as np
import torch
import os # Added import
from datetime import datetime
from transformers import pipeline
from sklearn.metrics import confusion_matrix

def benchmark_inference(model_path, test_texts, num_runs=100):
    """
    Benchmark model inference performance.
    
    Args:
        model_path (str): Path to the saved model
        test_texts (list): List of texts to test
        num_runs (int): Number of inference runs for averaging
        
    Returns:
        dict: Benchmark results including timing and memory usage
    """
    logging.info("Running inference benchmark...")
    
    # Load model and tokenizer
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Measure inference time
    total_time = 0
    memory_usage = []
    
    for _ in range(num_runs):
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        _ = classifier(test_texts)
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time += (end_time - start_time)
        memory_usage.append(final_memory - initial_memory)
    
    avg_inference_time = total_time / num_runs
    avg_memory_usage = sum(memory_usage) / len(memory_usage)
    
    benchmark_results = {
        "average_inference_time_seconds": avg_inference_time,
        "average_memory_usage_mb": avg_memory_usage,
        "num_runs": num_runs,
        "num_test_samples": len(test_texts),
        "device": "GPU" if torch.cuda.is_available() else "CPU"
    }
    
    return benchmark_results

def compute_extended_metrics(pred):
    """
    Compute extended evaluation metrics including confusion matrix.
    
    Args:
        pred: Prediction object containing label_ids and predictions
        
    Returns:
        dict: Dictionary of computed metrics
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix.tolist()
    }

def measure_training_performance(trainer, train_func):
    """
    Measure performance metrics during model training.
    
    Args:
        trainer: Hugging Face trainer object
        train_func: Training function to wrap with measurements
        
    Returns:
        tuple: (Training result, performance metrics)
    """
    # Record start time and initial memory
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run training
    train_result = train_func()
    
    # Record end time and final memory
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    training_time = end_time - start_time
    memory_used = final_memory - initial_memory
    
    # Log performance metrics
    logging.info(f"Training time: {training_time:.2f} seconds")
    logging.info(f"Memory used: {memory_used:.2f} MB")
    
    # Add metrics to training results
    performance_metrics = {
        "training_time": training_time,
        "memory_used_mb": memory_used,
        "gpu_used": torch.cuda.is_available()
    }
    
    return train_result, performance_metrics

def save_benchmark_results(output_dir, training_metrics, eval_metrics, benchmark_results, model_config):
    """
    Save all benchmark results to a JSON file.
    
    Args:
        output_dir (str): Directory to save results
        training_metrics (dict): Metrics from training
        eval_metrics (dict): Metrics from evaluation
        benchmark_results (dict): Results from benchmarking
        model_config (dict): Model configuration parameters
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    benchmark_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    
    all_metrics = {
        "timestamp": timestamp,
        "model_config": model_config,
        "training_performance": training_metrics, # Renamed for clarity
        "evaluation_metrics": eval_metrics,
        "inference_benchmark": benchmark_results # Renamed for clarity
    }
    
    try:
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=4) # Use indent for readability
        logging.info(f"Benchmark results saved successfully to: {benchmark_path}")
    except IOError as e:
        logging.error(f"Error saving benchmark results to {benchmark_path}: {e}")
    except TypeError as e:
        logging.error(f"Error serializing benchmark results to JSON: {e}. Check data types.")