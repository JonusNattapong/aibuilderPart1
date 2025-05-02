import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
from Dataset.fact_extraction_dataset import fact_extraction_data
import re

class FactExtractor:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        
        # Define fact patterns
        self.patterns = {
            'personal': {
                'name': r'(นาย|นางสาว|นาง)\s*([^\n\d]+)',
                'birth_date': r'เกิดเมื่อวันที่\s*(\d+\s+[^\d\n]+\s+\d+)',
                'address': r'อยู่ที่([^\n]+)',
                'occupation': r'ทำงานเป็น([^\n]+)',
                'education': r'จบการศึกษา([^\n]+)',
                'company': r'บริษัท([^\n]+)จำกัด'
            },
            'event': {
                'event_name': r'([^\n]+)\s*จะจัดขึ้น|จัดขึ้น',
                'date': r'วันที่\s*(\d+\s+[^\d\n]+\s+\d+)',
                'time': r'เวลา\s*(\d+[:\.]\d+\s*น\.)',
                'location': r'ณ\s*([^\n]+)',
                'price': r'ราคา\s*(\d+[\d,]*\s*บาท)'
            },
            'business': {
                'company_name': r'บริษัท([^\n]+)จำกัด',
                'profit': r'กำไรสุทธิ\s*(\d+[\d,]*\s*ล้านบาท)',
                'growth': r'เพิ่มขึ้น\s*(\d+%)',
                'revenue_source': r'รายได้หลักมาจาก([^\n]+)'
            }
        }
    
    def extract_structured_facts(self, text):
        facts = {}
        
        # Determine text type
        if any(marker in text.lower() for marker in ['บริษัท', 'กำไร', 'ธุรกิจ']):
            pattern_type = 'business'
        elif any(marker in text.lower() for marker in ['จัดขึ้น', 'งาน', 'กิจกรรม']):
            pattern_type = 'event'
        else:
            pattern_type = 'personal'
        
        # Extract facts using patterns
        for fact_name, pattern in self.patterns[pattern_type].items():
            matches = re.search(pattern, text)
            if matches:
                facts[fact_name] = matches.group(1).strip()
        
        return facts
    
    def format_facts(self, facts):
        formatted = []
        for key, value in facts.items():
            formatted.append(f"{key}: {value}")
        return "\n".join(formatted)
    
    def generate_fact_summary(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            f"สรุปข้อเท็จจริงจากข้อความต่อไปนี้: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate summary
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model():
    extractor = FactExtractor()
    results = []
    
    for item in fact_extraction_data:
        # Extract facts using patterns
        extracted_facts = extractor.extract_structured_facts(item['text'])
        formatted_facts = extractor.format_facts(extracted_facts)
        
        # Generate summary
        summary = extractor.generate_fact_summary(item['text'])
        
        results.append({
            'text': item['text'],
            'extracted_facts': formatted_facts,
            'generated_summary': summary,
            'expected_facts': item['extracted_facts']
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการสกัดข้อเท็จจริง:\n")
    
    results = evaluate_model()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print(f"ข้อความต้นฉบับ:")
        print(result['text'])
        print("\nข้อเท็จจริงที่สกัดได้ (แบบ Pattern Matching):")
        print(result['extracted_facts'])
        print("\nข้อเท็จจริงที่คาดหวัง:")
        print(result['expected_facts'])
        print("\nสรุปที่สร้างโดยโมเดล:")
        print(result['generated_summary'])
        print("-" * 50)

def run_interactive_demo():
    extractor = FactExtractor()
    
    print("\nทดสอบการสกัดข้อเท็จจริงแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการสกัดข้อเท็จจริง (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        # Extract facts
        facts = extractor.extract_structured_facts(text)
        formatted_facts = extractor.format_facts(facts)
        
        # Generate summary
        summary = extractor.generate_fact_summary(text)
        
        print("\nข้อเท็จจริงที่สกัดได้:")
        print(formatted_facts)
        print("\nสรุปข้อเท็จจริง:")
        print(summary)

def analyze_text_types():
    extractor = FactExtractor()
    
    print("\nการวิเคราะห์ประเภทข้อความต่างๆ:")
    
    example_texts = {
        "ข้อมูลส่วนบุคคล": "นายวิชัย ใจเย็น อายุ 35 ปี ทำงานเป็นแพทย์ที่โรงพยาบาลรามา",
        "ข้อมูลกิจกรรม": "งานสัมมนาวิชาการจะจัดขึ้นในวันที่ 1 มิถุนายน 2567 ที่โรงแรมเซ็นทารา",
        "ข้อมูลธุรกิจ": "บริษัทไทยฟู้ดส์จำกัด รายงานยอดขายเพิ่มขึ้น 20% ในไตรมาสล่าสุด"
    }
    
    for text_type, text in example_texts.items():
        print(f"\nประเภท: {text_type}")
        print(f"ข้อความ: {text}")
        
        # Extract facts
        facts = extractor.extract_structured_facts(text)
        formatted_facts = extractor.format_facts(facts)
        
        print("\nข้อเท็จจริงที่สกัดได้:")
        print(formatted_facts)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลสกัดข้อเท็จจริง...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze different text types
    analyze_text_types()