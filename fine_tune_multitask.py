# -*- coding: utf-8 -*-
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import os

# --- 1. Load Data ---
# Assuming datasets are in a 'Dataset' subdirectory relative to this script
dataset_dir = os.path.join(os.path.dirname(__file__), 'Dataset')
import sys
sys.path.append(dataset_dir)

try:
    from medical_dataset import medical_domain_data
    from finance_dataset import finance_domain_data
    from retail_dataset import retail_domain_data
    from legal_dataset import legal_domain_data
    print("Successfully imported datasets.")
except ImportError as e:
    print(f"Error importing datasets: {e}")
    print("Please ensure medical_dataset.py, finance_dataset.py, retail_dataset.py, and legal_dataset.py exist in the 'Dataset' directory.")
    sys.exit(1)

# --- 2. Combine and Preprocess Data ---
all_data = []
domain_map = {
    "medical": medical_domain_data,
    "finance": finance_domain_data,
    "retail": retail_domain_data,
    "legal": legal_domain_data,
}

def preprocess_data(domain_name, task_name, data_list):
    processed = []
    for item in data_list:
        input_text = ""
        target_text = ""
        prefix = f"{domain_name} {task_name}: "

        if task_name == "summarization":
            input_text = prefix + item.get("document", "")
            target_text = item.get("summary", "")
        elif task_name == "open_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer", "")
        elif task_name == "close_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer_text", "")
        elif task_name == "classification":
            input_text = prefix + item.get("text", "")
            target_text = item.get("label", "")
        elif task_name == "creative_writing":
            input_text = prefix + item.get("prompt", "")
            target_text = item.get("generated_text", "")
        elif task_name == "brainstorming":
            input_text = prefix + item.get("topic", "")
            target_text = "\n".join(item.get("ideas", []))
        elif task_name == "multiple_choice_qa":
            choices_str = " | ".join(item.get("choices", []))
            input_text = prefix + f"question: {item.get('question', '')} choices: {choices_str} context: {item.get('context', '')}"
            answer_idx = item.get("answer_index")
            if answer_idx is not None and item.get("choices"):
                target_text = item["choices"][answer_idx]

        if input_text and target_text:
            processed.append({"input_text": input_text, "target_text": target_text})
    return processed

print("Preprocessing data...")
for domain, data in domain_map.items():
    for task, task_data in data.items():
        all_data.extend(preprocess_data(domain, task, task_data))

print(f"Total processed data points: {len(all_data)}")
if not all_data:
    print("No data was processed. Check dataset files and preprocessing logic.")
    sys.exit(1)

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_list(all_data)
# Optional: Split dataset if needed (e.g., train/validation)
# hf_dataset = hf_dataset.train_test_split(test_size=0.1)
# train_dataset = hf_dataset["train"]
# eval_dataset = hf_dataset["test"]
train_dataset = hf_dataset # Using all data for training for simplicity

# --- 3. Load Model and Tokenizer ---
model_name = "google/mt5-small" # Example model
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
    model_inputs = tokenizer(examples["input_text"], max_length=max_input_length, truncation=True, padding="max_length")
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
print("Tokenization complete.")

# --- 5. Set Training Arguments ---
output_dir = "./results_multitask_finetune"
training_args = Seq2SeqTrainingArguments(
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
    logging_dir='./logs',
    logging_steps=10,
    # Add push_to_hub=True if you want to upload to Hugging Face Hub
)

# --- 6. Initialize Trainer ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    # eval_dataset=tokenized_datasets["test"], # Uncomment if you have eval split
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 7. Start Training ---
print("Starting training...")
try:
    trainer.train()
    print("Training finished.")
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)


# --- 8. Save Model ---
print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model and tokenizer saved.")

# --- Optional: Example Inference ---
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
