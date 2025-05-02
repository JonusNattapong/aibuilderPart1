import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from DatasetNLP.text2text_generation_nlp_dataset import text2text_data
import pandas as pd

class Text2TextGenerator:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        
        # Define task templates
        self.task_templates = {
            "simplify": "แปลงเป็นข้อความที่เข้าใจง่าย: ",
            "correct": "แก้ไขไวยากรณ์: ",
            "to_informal": "แปลงเป็นภาษาไม่ทางการ: ",
            "generate_question": "สร้างคำถามจากข้อความ: ",
            "extract_keywords": "สกัดคำสำคัญ: "
        }
    
    def generate(self, input_text, max_length=100, num_return_sequences=1, 
                temperature=0.7, top_k=50, top_p=0.9):
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Generate text
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode and return generated texts
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def process_task(self, task_type, text):
        # Get task template
        template = self.task_templates.get(task_type, "")
        if not template:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Prepare input with template
        input_text = template + text
        
        # Generate output
        return self.generate(input_text)[0]

def evaluate_model():
    generator = Text2TextGenerator()
    results = []
    
    for example in text2text_data:
        # Extract task type from input
        task_type = example['input_text'].split(":")[0]
        text = ":".join(example['input_text'].split(":")[1:]).strip()
        
        # Generate output
        generated_text = generator.process_task(task_type, text)
        
        results.append({
            'task': task_type,
            'input': text,
            'generated': generated_text,
            'target': example['target_text'],
            'matches_target': generated_text.lower() == example['target_text'].lower()
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการแปลงข้อความตามประเภทงาน:\n")
    
    results = evaluate_model()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print(f"ประเภทงาน: {result['task']}")
        print(f"ข้อความต้นฉบับ: {result['input']}")
        print(f"ข้อความที่สร้าง: {result['generated']}")
        print(f"ข้อความเป้าหมาย: {result['target']}")
        print(f"ตรงกับเป้าหมาย: {'ใช่' if result['matches_target'] else 'ไม่ใช่'}")
        print("-" * 50)
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r['matches_target']) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    generator = Text2TextGenerator()
    
    print("\nทดสอบการแปลงข้อความแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nเลือกประเภทงาน:")
        for task in generator.task_templates.keys():
            print(f"- {task}")
        
        task = input("\nใส่ประเภทงาน (หรือพิมพ์ 'exit' เพื่อออก): ")
        if task.lower() == 'exit':
            break
        
        if task not in generator.task_templates:
            print("ประเภทงานไม่ถูกต้อง")
            continue
        
        print("\nใส่ข้อความที่ต้องการแปลง:")
        text = input()
        
        print("\nกำลังสร้างข้อความ...")
        try:
            generated_texts = generator.process_task(task, text)
            print(f"\nผลลัพธ์: {generated_texts}")
        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {str(e)}")

def demo_multiple_outputs():
    generator = Text2TextGenerator()
    
    print("\nตัวอย่างการสร้างหลายรูปแบบ:")
    
    text = "เทคโนโลยี AI กำลังเปลี่ยนแปลงการทำงานในหลายอุตสาหกรรม"
    
    print(f"\nข้อความต้นฉบับ: {text}")
    
    # Generate multiple variations with different parameters
    print("\n1. การสร้างแบบหลายรูปแบบ (temperature=0.9):")
    outputs = generator.generate(
        generator.task_templates["simplify"] + text,
        num_return_sequences=3,
        temperature=0.9
    )
    for i, output in enumerate(outputs, 1):
        print(f"รูปแบบที่ {i}: {output}")
    
    print("\n2. การสร้างแบบควบคุม (temperature=0.3):")
    outputs = generator.generate(
        generator.task_templates["simplify"] + text,
        num_return_sequences=3,
        temperature=0.3
    )
    for i, output in enumerate(outputs, 1):
        print(f"รูปแบบที่ {i}: {output}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลแปลงข้อความ...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Show multiple outputs demo
    demo_multiple_outputs()