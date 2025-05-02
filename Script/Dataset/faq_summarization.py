import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
from Dataset.faq_summarization_dataset import faq_data

class FAQSummarizer:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
    
    def generate_answer(self, source_document, question, max_length=128):
        # Prepare input by combining document and question
        input_text = f"คำถาม: {question}\n\nเอกสาร: {source_document}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Generate answer
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode and clean up the generated answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    
    def extract_key_points(self, text):
        # Split text into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Initialize categories
        key_points = {
            'requirements': [],
            'timeframes': [],
            'conditions': [],
            'contact_info': []
        }
        
        # Keywords for each category
        keywords = {
            'requirements': ['ต้อง', 'จำเป็น', 'กรุณา', 'โปรด'],
            'timeframes': ['วัน', 'เวลา', 'ภายใน', 'ระยะเวลา'],
            'conditions': ['หาก', 'กรณี', 'เงื่อนไข', 'ข้อกำหนด'],
            'contact_info': ['ติดต่อ', 'โทร', 'อีเมล', 'เบอร์']
        }
        
        # Categorize sentences
        for sentence in sentences:
            for category, words in keywords.items():
                if any(word in sentence for word in words):
                    key_points[category].append(sentence)
        
        return key_points

def evaluate_model():
    summarizer = FAQSummarizer()
    results = []
    
    for item in faq_data:
        # Generate answer
        generated_answer = summarizer.generate_answer(
            item['source_document'],
            item['faq_question']
        )
        
        # Extract key points from source document
        key_points = summarizer.extract_key_points(item['source_document'])
        
        results.append({
            'source_document': item['source_document'],
            'question': item['faq_question'],
            'expected_answer': item['faq_answer'],
            'generated_answer': generated_answer,
            'key_points': key_points
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการสรุปคำถาม-คำตอบ FAQ:\n")
    
    results = evaluate_model()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print(f"เอกสารต้นฉบับ:")
        print(result['source_document'])
        print(f"\nคำถาม: {result['question']}")
        print(f"คำตอบที่คาดหวัง: {result['expected_answer']}")
        print(f"คำตอบที่สร้าง: {result['generated_answer']}")
        
        print("\nประเด็นสำคัญที่พบ:")
        for category, points in result['key_points'].items():
            if points:
                print(f"\n{category}:")
                for point in points:
                    print(f"- {point}")
        print("-" * 50)

def run_interactive_demo():
    summarizer = FAQSummarizer()
    
    print("\nทดสอบการสรุป FAQ แบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่เอกสารต้นฉบับ (หรือพิมพ์ 'exit' เพื่อออก):")
        document = input()
        if document.lower() == 'exit':
            break
        
        print("\nใส่คำถาม:")
        question = input()
        
        # Generate answer
        answer = summarizer.generate_answer(document, question)
        
        # Extract key points
        key_points = summarizer.extract_key_points(document)
        
        print("\nคำตอบที่สร้าง:")
        print(answer)
        
        print("\nประเด็นสำคัญที่พบในเอกสาร:")
        for category, points in key_points.items():
            if points:
                print(f"\n{category}:")
                for point in points:
                    print(f"- {point}")

def analyze_faq_patterns():
    summarizer = FAQSummarizer()
    
    print("\nการวิเคราะห์รูปแบบ FAQ ต่างๆ:")
    
    example_faqs = {
        "นโยบายการคืนสินค้า": {
            "document": "ลูกค้าสามารถคืนสินค้าได้ภายใน 30 วันหลังจากได้รับสินค้า โดยสินค้าต้องอยู่ในสภาพสมบูรณ์ ไม่มีร่องรอยการใช้งาน",
            "questions": [
                "ระยะเวลาในการคืนสินค้า",
                "เงื่อนไขการคืนสินค้า"
            ]
        },
        "การรับประกัน": {
            "document": "สินค้ารับประกัน 1 ปีเต็ม ครอบคลุมความเสียหายจากการผลิต ไม่รวมความเสียหายจากการใช้งานผิดวิธี",
            "questions": [
                "ระยะเวลารับประกัน",
                "การรับประกันครอบคลุมอะไรบ้าง"
            ]
        }
    }
    
    for category, data in example_faqs.items():
        print(f"\nหมวด: {category}")
        print(f"เอกสาร: {data['document']}")
        
        for question in data['questions']:
            print(f"\nคำถาม: {question}")
            answer = summarizer.generate_answer(data['document'], question)
            print(f"คำตอบ: {answer}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลสรุป FAQ...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze FAQ patterns
    analyze_faq_patterns()