import argparse
import pandas as pd
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from safetensors.torch import save_model

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a text classification model.')
    parser.add_argument('--dataset_path', type=str, default='Script/Generate/OUTPUT_DIR/text_classification_dataset.csv', help='Path to the dataset CSV file.')
    parser.add_argument('--model_name', type=str, default='Script/Models/wangchanberta-base-att-spm-uncased', help='Path to the pre-trained model directory.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--model_dir', type=str, default='Script/Models/text_classification_model', help='Directory to save the fine-tuned model.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the dataset
    df = pd.read_csv(args.dataset_path)

    # Split the data into training and validation sets
    train_text, val_text, train_labels, val_labels = train_test_split(df['text'], df['label'], random_state=42, test_size=0.2)

    # Define a custom dataset class
    class TextClassificationDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = self.texts[item]
            label = self.labels[item]

            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(df['label'].unique()))

    # Create dataset and data loader
    dataset_train = TextClassificationDataset(train_text, train_labels, tokenizer, max_len=512)
    dataset_val = TextClassificationDataset(val_text, val_labels, tokenizer, max_len=512)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size)

    # Fine-tune the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in data_loader_train:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader_train)}')

        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in data_loader_val:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, dim=1)

                predictions.extend(predicted.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())

        accuracy = accuracy_score(labels, predictions)
        print(f'Epoch {epoch+1}, Val Acc: {accuracy:.4f}')

    # Create a directory for the fine-tuned model
    os.makedirs(args.model_dir, exist_ok=True)

    save_model(model, os.path.join(args.model_dir, "model.safetensors"))
    # Save additional model components as separate safetensors files
    for name, module in model.named_modules():
        if hasattr(module, 'state_dict'):
            save_model(module, os.path.join(args.model_dir, f"{name}.safetensors"))

    # Save the fine-tuned model
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

if __name__ == "__main__":
    main()