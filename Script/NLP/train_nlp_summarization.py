import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from DatasetNLP.summarization_nlp_dataset import summarization_data

def prepare_data():
    # Convert to DataFrame
    df = pd.DataFrame(summarization_data)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df

def create_dataset(df, tokenizer, max_input_length=512, max_target_length=128):
    # Tokenize inputs and targets
    def preprocess_function(examples):
        # Tokenize documents
        model_inputs = tokenizer(
            examples['document'],
            max_length=max_input_length,
            padding='max_length',
            truncation=True
        )
        
        # Tokenize summaries
        labels = tokenizer(
            examples['summary'],
            max_length=max_target_length,
            padding='max_length',
            truncation=True
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Convert to datasets format
    dataset = Dataset.from_pandas(df)
    
    # Apply preprocessing
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        batch_size=8,
        num_proc=1
    )
    
    return tokenized_dataset

def compute_metrics(pred):
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    predictions = pred.predictions
    labels = pred.label_ids
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate ROUGE scores
    scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0
    }
    
    for pred, label in zip(decoded_preds, decoded_labels):
        results = scorer.score(label, pred)
        scores['rouge1'] += results['rouge1'].fmeasure
        scores['rouge2'] += results['rouge2'].fmeasure
        scores['rougeL'] += results['rougeL'].fmeasure
    
    # Calculate average scores
    num_samples = len(decoded_preds)
    for key in scores:
        scores[key] = scores[key] / num_samples
    
    return scores

def train_model():
    global tokenizer  # Make tokenizer accessible in compute_metrics
    
    # Initialize tokenizer and model
    model_name = "airesearch/wangchanberta-base-att-spm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Prepare data
    train_df, val_df = prepare_data()
    
    # Create datasets
    train_dataset = create_dataset(train_df, tokenizer)
    val_dataset = create_dataset(val_df, tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./summarization_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./summarization_logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        predict_with_generate=True
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    model.save_pretrained("./summarization_model")
    tokenizer.save_pretrained("./summarization_model")

def generate_summary(text, model_path="./summarization_model", max_length=128):
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    # Train model
    print("เริ่มการเทรนโมเดล...")
    train_model()
    print("เทรนโมเดลเสร็จสิ้น")
    
    # Test generation
    test_text = """
    การประชุมสุดยอดอาเซียนครั้งที่ 40 จัดขึ้นที่กรุงเทพมหานคร 
    มีการหารือประเด็นสำคัญเกี่ยวกับความร่วมมือทางเศรษฐกิจและความมั่นคงในภูมิภาค 
    โดยมีผู้นำจาก 10 ประเทศสมาชิกเข้าร่วม พร้อมทั้งมีการลงนามข้อตกลงความร่วมมือหลายฉบับ
    """
    summary = generate_summary(test_text)
    print(f"\nทดสอบการสรุปความ:")
    print(f"ข้อความต้นฉบับ: {test_text}")
    print(f"สรุป: {summary}")