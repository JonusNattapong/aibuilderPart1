import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
from DatasetNLP.qa_nlp_dataset import qa_data

class QuestionAnswerer:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
    
    def answer_question(self, question, context):
        # Tokenize question and context
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the most likely beginning and end of answer
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Get the most likely answer span
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        
        # Convert tokens to answer text
        answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
        answer = self.tokenizer.decode(answer_tokens)
        
        # Clean up answer
        answer = answer.strip()
        answer = answer.replace(self.tokenizer.pad_token, "").strip()
        
        # Calculate confidence score
        start_prob = torch.softmax(start_scores, dim=1)[0][start_index].item()
        end_prob = torch.softmax(end_scores, dim=1)[0][end_index].item()
        confidence = (start_prob + end_prob) / 2
        
        return {
            'answer': answer,
            'confidence': confidence,
            'start_index': start_index.item(),
            'end_index': end_index.item()
        }

def evaluate_model(qa_model, test_data):
    results = []
    
    for item in test_data:
        prediction = qa_model.answer_question(item['question'], item['context'])
        
        # Get ground truth answer
        true_answer = item['answers']['text'][0]
        
        # Calculate exact match (case-insensitive)
        is_exact_match = prediction['answer'].lower() == true_answer.lower()
        
        results.append({
            'id': item['id'],
            'question': item['question'],
            'context': item['context'],
            'predicted_answer': prediction['answer'],
            'true_answer': true_answer,
            'confidence': prediction['confidence'],
            'is_exact_match': is_exact_match
        })
    
    return pd.DataFrame(results)

def demonstrate_usage():
    # Initialize model
    qa_model = QuestionAnswerer()
    
    # Evaluate on test data
    print("ทดสอบโมเดลกับชุดข้อมูลทดสอบ:")
    results_df = evaluate_model(qa_model, qa_data)
    
    # Print results
    print("\nผลการทดสอบ:")
    for _, row in results_df.iterrows():
        print(f"\nID: {row['id']}")
        print(f"คำถาม: {row['question']}")
        print(f"บริบท: {row['context']}")
        print(f"คำตอบที่ทำนาย: {row['predicted_answer']}")
        print(f"คำตอบที่ถูกต้อง: {row['true_answer']}")
        print(f"ความมั่นใจ: {row['confidence']:.4f}")
        print(f"ตรงกับคำตอบที่ถูกต้อง: {'ใช่' if row['is_exact_match'] else 'ไม่ใช่'}")
    
    # Calculate and print metrics
    accuracy = results_df['is_exact_match'].mean()
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    qa_model = QuestionAnswerer()
    
    print("\nทดสอบการถาม-ตอบแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        # Get context
        print("\nใส่บริบท (หรือพิมพ์ 'exit' เพื่อออก):")
        context = input()
        if context.lower() == 'exit':
            break
        
        while True:
            # Get question
            print("\nใส่คำถาม (หรือพิมพ์ 'new' เพื่อเปลี่ยนบริบท, 'exit' เพื่อออก):")
            question = input()
            if question.lower() == 'exit':
                return
            if question.lower() == 'new':
                break
            
            # Get answer
            prediction = qa_model.answer_question(question, context)
            
            print(f"\nคำตอบ: {prediction['answer']}")
            print(f"ความมั่นใจ: {prediction['confidence']:.4f}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลถาม-ตอบ...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()