import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from DatasetNLP.table_qa_nlp_dataset import table_qa_data

class TableQuestionAnswerer:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
    
    def prepare_table_context(self, table):
        # Convert table to string format
        header = " | ".join(table['header'])
        rows = [" | ".join(str(cell) for cell in row) for row in table['rows']]
        return "ส่วนหัว: " + header + "\n" + "\n".join(rows)
    
    def answer_question(self, table, question):
        # Convert table to string context
        context = self.prepare_table_context(table)
        
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
            'confidence': confidence
        }

def find_value_in_table(table, target_value):
    # Search for the target value in the table
    for row_idx, row in enumerate(table['rows']):
        for col_idx, cell in enumerate(row):
            if str(cell) == str(target_value):
                return [row_idx, col_idx]
    return None

def evaluate_model():
    qa_model = TableQuestionAnswerer()
    results = []
    
    for example in table_qa_data:
        # Get model prediction
        prediction = qa_model.answer_question(example['table'], example['question'])
        
        # Create DataFrame for display
        df = pd.DataFrame(example['table']['rows'], columns=example['table']['header'])
        
        # Find predicted answer coordinates
        pred_coordinates = find_value_in_table(example['table'], prediction['answer'])
        
        results.append({
            'table': df,
            'question': example['question'],
            'true_answer': example['answer_text'],
            'predicted_answer': prediction['answer'],
            'confidence': prediction['confidence'],
            'is_correct': prediction['answer'] == example['answer_text']
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการถาม-ตอบข้อมูลในตาราง:\n")
    
    results = evaluate_model()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print("\nตาราง:")
        print(result['table'])
        print(f"\nคำถาม: {result['question']}")
        print(f"คำตอบที่ถูกต้อง: {result['true_answer']}")
        print(f"คำตอบที่ทำนาย: {result['predicted_answer']}")
        print(f"ความมั่นใจ: {result['confidence']:.4f}")
        print(f"ถูกต้อง: {'ใช่' if result['is_correct'] else 'ไม่ใช่'}")
        print("-" * 50)
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r['is_correct']) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    qa_model = TableQuestionAnswerer()
    
    print("\nทดสอบการถาม-ตอบตารางแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        # Create a sample table or let user input data
        print("\nเลือกตาราง:")
        for i, example in enumerate(table_qa_data, 1):
            print(f"\n{i}. ตารางที่ {i}:")
            df = pd.DataFrame(example['table']['rows'], columns=example['table']['header'])
            print(df)
        
        choice = input("\nเลือกตารางหมายเลข (หรือพิมพ์ 'exit' เพื่อออก): ")
        if choice.lower() == 'exit':
            break
        
        try:
            table_idx = int(choice) - 1
            if not (0 <= table_idx < len(table_qa_data)):
                print("กรุณาเลือกหมายเลขตารางที่ถูกต้อง")
                continue
        except ValueError:
            print("กรุณาป้อนหมายเลขที่ถูกต้อง")
            continue
        
        while True:
            print("\nใส่คำถาม (หรือพิมพ์ 'new' เพื่อเปลี่ยนตาราง, 'exit' เพื่อออก):")
            question = input()
            
            if question.lower() == 'exit':
                return
            if question.lower() == 'new':
                break
            
            # Get answer
            result = qa_model.answer_question(table_qa_data[table_idx]['table'], question)
            
            print(f"\nคำตอบ: {result['answer']}")
            print(f"ความมั่นใจ: {result['confidence']:.4f}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลถาม-ตอบตาราง...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()