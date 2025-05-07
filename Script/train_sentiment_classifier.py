import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, classification_report
import argparse

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def load_data(data_dir):
    """Load train/val/test splits and label mapping"""
    # Load label mapping
    with open(os.path.join(data_dir, 'label_map.json'), 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    # Load splits
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Convert labels to ids
    train_labels = train_df['label'].map(label_map).values
    val_labels = val_df['label'].map(label_map).values
    test_labels = test_df['label'].map(label_map).values
    
    return (
        train_df['text'].tolist(), train_labels,
        val_df['text'].tolist(), val_labels,
        test_df['text'].tolist(), test_labels,
        label_map
    )

def train_model(args):
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, label_map = load_data(args.data_dir)
    num_labels = len(label_map)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # Convert numeric labels back to text labels
    label_map_reverse = {v: k for k, v in label_map.items()}
    pred_labels_text = [label_map_reverse[i] for i in pred_labels]
    true_labels_text = [label_map_reverse[i] for i in test_labels]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels_text, pred_labels_text))
    
    # Save model and tokenizer
    final_model_path = os.path.join(args.output_dir, 'final')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nModel saved to: {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train sentiment classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing train/val/test splits')
    parser.add_argument('--model_name', type=str, 
                      default='microsoft/mdeberta-v3-base',
                      help='Pretrained model to fine-tune')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for model and checkpoints')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_model(args)

if __name__ == '__main__':
    main()