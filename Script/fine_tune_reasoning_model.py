import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM
)
from sklearn.model_selection import train_test_split
from config_generate_reasoning import (
    REASONING_TASKS,
    OUTPUT_DIR
)

class ReasoningDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_data(task, data_dir):
    df = pd.read_csv(os.path.join(data_dir, f"{task}.csv"))
    
    if task == 'chain_of_thought':
        texts = df['question'].tolist()
        labels = df['answer'].tolist()
    elif task == 'meta_reasoning':
        texts = df['question'].tolist()
        labels = df['strategy'].tolist()
    elif task == 'pattern_recognition':
        texts = df['sequence'].tolist()
        labels = df['next_value'].tolist()
    elif task == 'react':
        texts = df['situation'].tolist()
        labels = df['actions'].tolist()
    elif task == 'reflection':
        texts = df['experience'].tolist()
        labels = df['lessons_learned'].tolist()
    elif task == 'toolformer':
        texts = df['problem'].tolist()
        labels = df['selected_tool'].tolist()
    
    return texts, labels

def train_model(args):
    # Load data
    texts, labels = prepare_data(args.task, args.data_dir)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_type == 'causal':
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(set(labels))
        )

    # Create datasets
    train_dataset = ReasoningDataset(train_texts, train_labels, tokenizer)
    val_dataset = ReasoningDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"models/{args.task}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"logs/{args.task}",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(f"models/{args.task}/final")
    tokenizer.save_pretrained(f"models/{args.task}/final")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune models for reasoning tasks')
    parser.add_argument('--task', type=str, required=True, choices=REASONING_TASKS.keys(),
                        help='Reasoning task to train model for')
    parser.add_argument('--data_dir', type=str, default=OUTPUT_DIR,
                        help='Directory containing the task datasets')
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2',
                        help='Pretrained model to fine-tune')
    parser.add_argument('--model_type', type=str, default='causal',
                        choices=['causal', 'sequence'],
                        help='Type of model architecture')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    args = parser.parse_args()

    train_model(args)

if __name__ == "__main__":
    main()