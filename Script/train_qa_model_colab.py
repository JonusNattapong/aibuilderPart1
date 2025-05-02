# -*- coding: utf-8 -*-
"""
train_qa_model_colab.py

This script trains a Question Answering model using the Hugging Face
Transformers library on the generated Thai QA dataset.
It is adapted from the Colab notebook format for local execution
(or execution on a server with necessary libraries installed).
"""

# Import libraries
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Define the base path to your project directory
# !! IMPORTANT: Adjust this path if running outside the original project structure !!
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes script is in Script folder

DATASET_PATH = os.path.join(BASE_PATH, 'DataOutput/thai_dataset_qa.csv')
MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, 'Models/qa_model')
LOGGING_DIR = os.path.join(BASE_PATH, 'Script/logs') # Logging directory within Script folder

# Model checkpoint for tokenizer and base model
MODEL_CHECKPOINT = "airesearch/wangchanberta-base-att-spm-uncased" # Example Thai model

# Tokenization parameters
MAX_LENGTH = 384 # The maximum length of a feature (question and context)
DOC_STRIDE = 128 # The authorized overlap between two part of the context when splitting it is needed.

# Training parameters
LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 8   # Adjust based on GPU memory
EVAL_BATCH_SIZE = 8    # Adjust based on GPU memory
NUM_EPOCHS = 3         # Number of training epochs (adjust as needed)
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_loss"
FP16 = torch.cuda.is_available() # Use mixed precision training if GPU is available

# --- End Configuration ---

def load_and_prepare_dataset(dataset_path):
    """Loads the dataset from CSV and prepares it for Hugging Face."""
    logging.info(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        logging.info("Dataset loaded successfully.")
        logging.info(f"Initial dataset shape: {df.shape}")

        # Convert answer_start to integer, handle potential errors/missing values
        df['answer_start'] = pd.to_numeric(df['answer_start'], errors='coerce').fillna(-1).astype(int)

        # Filter out rows where answer_start is -1 (answer not found in context)
        initial_rows = len(df)
        df = df[df['answer_start'] != -1].reset_index(drop=True)
        filtered_rows = len(df)
        logging.info(f"Filtered out {initial_rows - filtered_rows} rows with invalid answers.")
        logging.info(f"Dataset shape after filtering: {df.shape}")

        if filtered_rows == 0:
            logging.error("No valid examples remaining after filtering. Check dataset generation.")
            raise ValueError("No valid examples found in the dataset.")

        # Create the 'answers' column in the required format
        df['answers'] = df.apply(lambda row: {'text': [row['answer_text']], 'answer_start': [row['answer_start']]}, axis=1)

        # Select necessary columns
        df_final = df[['id', 'context', 'question', 'answers']]

        # Convert pandas DataFrame to Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(df_final)

        # Split into train/validation sets (90% train, 10% validation)
        train_test_split = hf_dataset.train_test_split(test_size=0.1, seed=42) # Added seed for reproducibility
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })

        logging.info("Dataset prepared for Hugging Face:")
        logging.info(dataset_dict)
        return dataset_dict

    except FileNotFoundError:
        logging.error(f"Error: Dataset file not found at {dataset_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during dataset loading/preparation: {e}")
        raise

def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    """Tokenizes the examples for training."""
    # Remove leading whitespace from questions
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Check boundaries carefully
            context_start_char = offsets[token_start_index][0]
            context_end_char = offsets[token_end_index][1]

            if not (context_start_char <= start_char and context_end_char >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Adjust token indices to match answer boundaries
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                     token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)


    return tokenized_examples

def main():
    """Main function to run the training pipeline."""

    # Create output and logging directories if they don't exist
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    logging.info(f"Model output directory: {MODEL_OUTPUT_DIR}")
    logging.info(f"Logging directory: {LOGGING_DIR}")

    # 1. Load and Prepare Dataset
    dataset_dict = load_and_prepare_dataset(DATASET_PATH)

    # 2. Tokenization
    logging.info(f"Loading tokenizer from: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    logging.info("Tokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH, "doc_stride": DOC_STRIDE}
    )
    logging.info("Tokenization complete.")
    logging.info(tokenized_datasets)

    # 3. Load Model
    logging.info(f"Loading pre-trained model from: {MODEL_CHECKPOINT}")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

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
        report_to="tensorboard", # Logs to TensorBoard by default if installed
        seed=42, # For reproducibility
    )

    # 5. Initialize Trainer
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        # data_collator=default_data_collator # Default works for QA
    )

    # 6. Start Training
    logging.info("Starting training...")
    try:
        train_result = trainer.train()
        logging.info("Training finished.")

        # Save training metrics
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state() # Saves optimizer, scheduler, etc.

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise

    # 7. Evaluate Model
    logging.info("Evaluating model...")
    try:
        eval_metrics = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_metrics}")

        # Save evaluation metrics
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")
        # Continue to saving even if evaluation fails

    # 8. Save Model
    logging.info(f"Saving the best model to {MODEL_OUTPUT_DIR}...")
    try:
        trainer.save_model(MODEL_OUTPUT_DIR) # Saves the best model if load_best_model_at_end=True
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        logging.info("Model and tokenizer saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred during model saving: {e}")
        raise

    logging.info("\n--- Training and Saving Complete ---")
    logging.info(f"Model saved in: {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

    # Optional: Add code here to test the trained model after training
    from transformers import pipeline # Uncomment this import
    try:
        logging.info("Testing the trained model with an example...")
        qa_pipeline = pipeline(
            "question-answering",
            model=MODEL_OUTPUT_DIR,
            tokenizer=MODEL_OUTPUT_DIR,
            device=0 if torch.cuda.is_available() else -1
        )
        # Replace with actual context and question from your data or new examples
        context_example = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย และเป็นศูนย์กลางทางเศรษฐกิจ การคมนาคม และวัฒนธรรม" # Example context
        question_example = "เมืองหลวงของประเทศไทยคืออะไร" # Example question
        result = qa_pipeline(question=question_example, context=context_example)
        logging.info(f"Test Question: {question_example}")
        logging.info(f"Test Context: {context_example}")
        logging.info(f"Test Result: {result}")
    except Exception as e:
        logging.error(f"Failed to test the pipeline after training: {e}")

