import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from DatasetNLP.text_classification_nlp_dataset import text_classification_data

def prepare_data():
    # Convert to DataFrame
    df = pd.DataFrame(text_classification_data)
    
    # Create label mapping
    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label_id'] = df['label'].map(label_mapping)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df, label_mapping

def create_dataset(df, tokenizer):
    # Tokenize texts
    encodings = tokenizer(df['text'].tolist(), 
                         truncation=True, 
                         padding=True, 
                         max_length=128)
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': df['label_id'].tolist()
    })
    return dataset

def train_model():
    # Initialize tokenizer and model
    model_name = "airesearch/wangchanberta-base-att-spm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare data
    train_df, val_df, label_mapping = prepare_data()
    num_labels = len(label_mapping)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # Create datasets
    train_dataset = create_dataset(train_df, tokenizer)
    val_dataset = create_dataset(val_df, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True
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
    
    # Save model and tokenizer
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    
    # Save label mapping
    import json
    with open('./model/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)

def predict_text(text, model_path="./model"):
    # Load model, tokenizer and label mapping
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open(f'{model_path}/label_mapping.json', 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    # Reverse label mapping
    id2label = {v: k for k, v in label_mapping.items()}
    
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Get prediction
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs).item()
    
    return id2label[predicted_class]

if __name__ == "__main__":
    # Train model
    print("เริ่มการเทรนโมเดล...")
    train_model()
    print("เทรนโมเดลเสร็จสิ้น")
    
    # Test prediction
    test_text = "นายกรัฐมนตรีประกาศนโยบายใหม่"
    predicted_label = predict_text(test_text)
    print(f"\nทดสอบการทำนาย:")
    print(f"ข้อความ: {test_text}")
    print(f"ประเภท: {predicted_label}")