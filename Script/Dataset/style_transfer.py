import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
from Dataset.style_transfer_dataset import style_transfer_data

class StyleTransfer:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        
        # Define style indicators
        self.style_indicators = {
            'formal': {
                'word_pairs': [
                    ('ครับ/ค่ะ', 'นะ'),
                    ('กรุณา', 'ช่วย'),
                    ('ข้าพเจ้า', 'ฉัน'),
                    ('ท่าน', 'คุณ'),
                    ('โปรด', 'ช่วย')
                ],
                'markers': ['ขอความกรุณา', 'ขออนุญาต', 'ขอเรียน']
            },
            'informal': {
                'word_pairs': [
                    ('นะ', 'ครับ/ค่ะ'),
                    ('เลย', 'ทีเดียว'),
                    ('จ้า', 'ค่ะ/ครับ'),
                    ('อะ', 'นะคะ/ครับ')
                ],
                'markers': ['น่ะ', 'อ่ะ', 'แป๊บ', 'นะจ๊ะ']
            }
        }
    
    def convert_style(self, text, target_style='informal'):
        # Prepare input
        prefix = "แปลงเป็นภาษา" + ("ทางการ: " if target_style == 'formal' else "ไม่ทางการ: ")
        input_text = prefix + text
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        
        # Generate text
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def analyze_style(self, text):
        # Count style indicators
        style_scores = {'formal': 0, 'informal': 0}
        
        for style, indicators in self.style_indicators.items():
            # Check word pairs
            for formal, informal in indicators['word_pairs']:
                if formal in text:
                    style_scores['formal'] += 1
                if informal in text:
                    style_scores['informal'] += 1
            
            # Check markers
            for marker in indicators['markers']:
                if marker in text:
                    style_scores[style] += 1
        
        # Determine style
        if style_scores['formal'] > style_scores['informal']:
            return 'formal'
        elif style_scores['informal'] > style_scores['formal']:
            return 'informal'
        else:
            return 'neutral'

def evaluate_model():
    converter = StyleTransfer()
    results = []
    
    for item in style_transfer_data:
        # Convert both ways
        informal_to_formal = converter.convert_style(item['informal_text'], 'formal')
        formal_to_informal = converter.convert_style(item['formal_text'], 'informal')
        
        # Analyze styles
        original_formal_style = converter.analyze_style(item['formal_text'])
        original_informal_style = converter.analyze_style(item['informal_text'])
        converted_formal_style = converter.analyze_style(informal_to_formal)
        converted_informal_style = converter.analyze_style(formal_to_informal)
        
        results.append({
            'original_formal': item['formal_text'],
            'original_informal': item['informal_text'],
            'converted_to_formal': informal_to_formal,
            'converted_to_informal': formal_to_informal,
            'style_analysis': {
                'original_formal': original_formal_style,
                'original_informal': original_informal_style,
                'converted_formal': converted_formal_style,
                'converted_informal': converted_informal_style
            }
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการแปลงรูปแบบภาษา:\n")
    
    results = evaluate_model()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print("-" * 50)
        print("ข้อความทางการ:")
        print(f"ต้นฉบับ: {result['original_formal']}")
        print(f"วิเคราะห์รูปแบบ: {result['style_analysis']['original_formal']}")
        print(f"แปลงเป็นไม่ทางการ: {result['converted_to_informal']}")
        print(f"วิเคราะห์รูปแบบ: {result['style_analysis']['converted_informal']}")
        
        print("\nข้อความไม่ทางการ:")
        print(f"ต้นฉบับ: {result['original_informal']}")
        print(f"วิเคราะห์รูปแบบ: {result['style_analysis']['original_informal']}")
        print(f"แปลงเป็นทางการ: {result['converted_to_formal']}")
        print(f"วิเคราะห์รูปแบบ: {result['style_analysis']['converted_formal']}")
        print("-" * 50)

def run_interactive_demo():
    converter = StyleTransfer()
    
    print("\nทดสอบการแปลงรูปแบบภาษาแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการแปลง (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        # Analyze original style
        original_style = converter.analyze_style(text)
        print(f"\nรูปแบบต้นฉบับ: {original_style}")
        
        # Convert to opposite style
        target_style = 'informal' if original_style == 'formal' else 'formal'
        converted = converter.convert_style(text, target_style)
        
        print(f"\nแปลงเป็นภาษา{target_style}:")
        print(converted)
        print(f"วิเคราะห์รูปแบบ: {converter.analyze_style(converted)}")

def analyze_style_patterns():
    converter = StyleTransfer()
    
    print("\nการวิเคราะห์รูปแบบภาษาต่างๆ:")
    
    example_texts = {
        "ทางการมาก": "ข้าพเจ้าขอความกรุณาท่านโปรดพิจารณาคำร้องนี้",
        "ทางการปานกลาง": "กรุณารอสักครู่นะคะ",
        "ไม่ทางการปานกลาง": "รอแป๊บนึงนะ",
        "ไม่ทางการมาก": "เดี๋ยวมาเล้ย รอแปป๊นิดนึงน้า"
    }
    
    for style_desc, text in example_texts.items():
        print(f"\nตัวอย่าง ({style_desc}):")
        print(f"ข้อความ: {text}")
        
        # Analyze style
        detected_style = converter.analyze_style(text)
        print(f"วิเคราะห์รูปแบบ: {detected_style}")
        
        # Convert to opposite style
        target_style = 'informal' if detected_style == 'formal' else 'formal'
        converted = converter.convert_style(text, target_style)
        print(f"แปลงเป็น{target_style}: {converted}")
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลแปลงรูปแบบภาษา...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze style patterns
    analyze_style_patterns()