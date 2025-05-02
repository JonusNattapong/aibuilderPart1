# -*- coding: utf-8 -*-
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_dataset, concatenate_datasets # Updated imports
import os
import sys

# --- 1. Load Data from Parquet Files ---
# Define the DataOutput directory relative to this script's location
script_dir = os.path.dirname(__file__)
data_output_dir = os.path.join(script_dir, 'DataOutput') # Changed base directory for datasets

parquet_files = {
    "medical": os.path.join(data_output_dir, "medical_data.parquet"),
    "finance": os.path.join(data_output_dir, "finance_data.parquet"),
    "retail": os.path.join(data_output_dir, "retail_data.parquet"),
    "legal": os.path.join(data_output_dir, "legal_data.parquet"),
}

loaded_datasets = []
print("Loading datasets from Parquet files in DataOutput...")
for domain, file_path in parquet_files.items():
    if os.path.exists(file_path):
        try:
            ds = load_dataset('parquet', data_files=file_path)['train'] # load_dataset returns a DatasetDict
            print(f"Loaded {len(ds)} records from {file_path}")
            loaded_datasets.append(ds)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            print("Please ensure the Parquet file exists and is valid in the 'DataOutput' directory.")
            print(f"You might need to run the corresponding dataset script (e.g., python Dataset/{domain}_dataset.py) first to generate it.")
            sys.exit(1)
    else:
        print(f"Error: Parquet file not found at {file_path}")
        print(f"Please generate it by running: python {os.path.join('Dataset', domain + '_dataset.py')}")
        sys.exit(1)

if not loaded_datasets:
    print("No datasets were loaded. Exiting.")
    sys.exit(1)

# Concatenate all datasets into one
print("Concatenating datasets...")
combined_dataset = concatenate_datasets(loaded_datasets)
print(f"Total combined records: {len(combined_dataset)}")
print(f"Dataset features: {combined_dataset.features}")

# Optional: Shuffle the combined dataset
combined_dataset = combined_dataset.shuffle(seed=42)

# --- 2. Preprocessing (Now handled during Parquet generation) ---
# The old preprocessing loop and function are removed.
# We assume the Parquet files have 'input_text' and 'target_text' columns.

# --- 3. Load Model and Tokenizer ---
model_name = "google/mt5-small" # Example model
# ... (rest of the model/tokenizer loading code remains the same) ...
print(f"Loading model and tokenizer: {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model and tokenizer loaded.")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    sys.exit(1)


# --- 4. Tokenize Dataset ---
max_input_length = 512
max_target_length = 128

def tokenize_function(examples):
    # Ensure columns exist
    if "input_text" not in examples or "target_text" not in examples:
        raise ValueError("Parquet files must contain 'input_text' and 'target_text' columns.")

    model_inputs = tokenizer(examples["input_text"], max_length=max_input_length, truncation=True, padding="max_length")
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
# Define columns to remove based on the loaded dataset features
columns_to_remove = list(combined_dataset.features.keys())
tokenized_datasets = combined_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=columns_to_remove # Remove original columns after tokenization
)
print("Tokenization complete.")
print(f"Tokenized dataset features: {tokenized_datasets.features}")


# --- 5. Set Training Arguments ---
output_dir = "./results_multitask_finetune_parquet" # Changed output dir name slightly
training_args = Seq2SeqTrainingArguments(
    # ... (training arguments remain the same) ...
    output_dir=output_dir,
    evaluation_strategy="no", # No evaluation dataset defined for simplicity
    learning_rate=2e-5,
    per_device_train_batch_size=4, # Adjust based on GPU memory
    per_device_eval_batch_size=4,  # Adjust based on GPU memory
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1, # Start with 1 epoch, increase as needed
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Use mixed precision if CUDA is available
    logging_dir='./logs_parquet',
    logging_steps=10,
)

# --- 6. Initialize Trainer ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets, # Use the tokenized dataset
    # eval_dataset=tokenized_datasets["test"], # Uncomment if you have eval split
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 7. Start Training ---
# ... (training code remains the same) ...
print("Starting training...")
try:
    trainer.train()
    print("Training finished.")
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)


# --- 8. Save Model ---
# ... (saving code remains the same) ...
print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model and tokenizer saved.")


# --- Optional: Example Inference ---
# ... (inference code remains the same) ...
print("\n--- Example Inference ---")
# Load the fine-tuned model
# model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
# tokenizer = AutoTokenizer.from_pretrained(output_dir)

# Example prompt (modify as needed)
test_prompt = "medical summarization: ผู้ป่วยชายอายุ 70 ปี มาด้วยอาการปวดท้องรุนแรง ตรวจร่างกายพบกดเจ็บที่ท้องด้านขวาบน อัลตราซาวด์พบถุงน้ำดีอักเสบ แพทย์วินิจฉัยว่าเป็นถุงน้ำดีอักเสบเฉียบพลัน"
print(f"Input: {test_prompt}")

try:
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device) # Ensure tensor is on the same device as model
    outputs = model.generate(**inputs, max_new_tokens=100) # Adjust max_new_tokens as needed
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Output: {decoded_output}")
except Exception as e:
    print(f"Error during inference: {e}")

print("\nScript finished.")
