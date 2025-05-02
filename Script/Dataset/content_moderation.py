import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from Dataset.content_moderation_dataset import moderation_data

class ContentModerator:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()
        
        self.label_map = {
            0: "เหมาะสม",
            1: "ไม่เหมาะสม"
        }
    
    def predict(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get prediction and confidence
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        return {
            'text': text,
            'label': self.label_map[prediction],
            'confidence': confidence,
            'is_inappropriate': prediction == 1,
            'inappropriate_prob': probabilities[0][1].item()
        }
    
    def batch_predict(self, texts):
        return [self.predict(text) for text in texts]

def evaluate_model():
    moderator = ContentModerator()
    results = []
    
    for item in moderation_data:
        prediction = moderator.predict(item['text'])
        prediction['expected_label'] = moderator.label_map[item['label']]
        prediction['is_correct'] = (prediction['is_inappropriate'] == (item['label'] == 1))
        results.append(prediction)
    
    return results

def demonstrate_usage():
    print("ทดสอบการตรวจสอบเนื้อหา:\n")
    
    results = evaluate_model()
    
    # Print results grouped by correctness
    print("ผลการทำนายที่ถูกต้อง:")
    print("-" * 50)
    correct_results = [r for r in results if r['is_correct']]
    for result in correct_results:
        print(f"ข้อความ: {result['text']}")
        print(f"การทำนาย: {result['label']} (ความมั่นใจ: {result['confidence']:.4f})")
        print(f"ค่าที่ถูกต้อง: {result['expected_label']}")
        print("-" * 50)
    
    print("\nผลการทำนายที่ไม่ถูกต้อง:")
    print("-" * 50)
    incorrect_results = [r for r in results if not r['is_correct']]
    for result in incorrect_results:
        print(f"ข้อความ: {result['text']}")
        print(f"การทำนาย: {result['label']} (ความมั่นใจ: {result['confidence']:.4f})")
        print(f"ค่าที่ถูกต้อง: {result['expected_label']}")
        print("-" * 50)
    
    # Calculate metrics
    accuracy = len(correct_results) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    moderator = ContentModerator()
    
    print("\nทดสอบการตรวจสอบเนื้อหาแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการตรวจสอบ (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        result = moderator.predict(text)
        
        print(f"\nผลการตรวจสอบ:")
        print(f"การจัดประเภท: {result['label']}")
        print(f"ความมั่นใจ: {result['confidence']:.4f}")
        print(f"ความน่าจะเป็นที่ไม่เหมาะสม: {result['inappropriate_prob']:.4f}")
        
        # แสดงคำเตือนถ้าเนื้อหาไม่เหมาะสม
        if result['is_inappropriate']:
            print("\nคำเตือน: เนื้อหานี้อาจไม่เหมาะสม")

def analyze_content_types():
    moderator = ContentModerator()
    
    print("\nการวิเคราะห์ประเภทเนื้อหาที่ไม่เหมาะสม:")
    
    # ตัวอย่างเนื้อหาแต่ละประเภท
    content_types = {
        "การใช้คำหยาบคาย": "ด่าด้วยคำหยาบคาย",
        "การคุกคาม": "จะตามไปทวงหนี้ถึงบ้าน",
        "สแปม": "รับสมัครงาน รายได้ดี ทักแชทมา",
        "การหลอกลวง": "ลงทุนกับเรารับผลตอบแทน 100% ใน 1 เดือน",
        "เนื้อหาอันตราย": "สอนวิธีทำระเบิดง่ายๆ",
    }
    
    for content_type, text in content_types.items():
        result = moderator.predict(text)
        print(f"\nประเภท: {content_type}")
        print(f"ตัวอย่างข้อความ: {text}")
        print(f"ผลการตรวจสอบ: {result['label']}")
        print(f"ความมั่นใจ: {result['confidence']:.4f}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลตรวจสอบเนื้อหา...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze different types of inappropriate content
    analyze_content_types()