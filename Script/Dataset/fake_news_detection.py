import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from Dataset.fake_news_detection_dataset import fake_news_data

class FakeNewsDetector:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()
        
        # Label mapping
        self.id2label = {0: "ข่าวจริง", 1: "ข่าวปลอม"}
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    def predict(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get prediction and confidence
        prediction = torch.argmax(probabilities, dim=1).item()
        
        return {
            'text': text,
            'prediction': self.id2label[prediction],
            'confidence': probabilities[0][prediction].item(),
            'fake_probability': probabilities[0][1].item(),
            'real_probability': probabilities[0][0].item()
        }
    
    def analyze_content(self, text):
        # ตรวจสอบลักษณะที่อาจบ่งชี้ว่าเป็นข่าวปลอม
        risk_indicators = {
            "sensational": (
                any(word in text.lower() for word in ["ด่วน!", "ด่วนที่สุด!", "สุดยอด!"]) or
                text.count("!") > 2
            ),
            "clickbait": (
                "แชร์ต่อ" in text.lower() or
                "กดแชร์" in text.lower() or
                "ห้ามพลาด" in text.lower()
            ),
            "unrealistic_claims": (
                any(word in text.lower() for word in ["100%", "ทุกคน", "ทั้งหมด", "หายขาด"]) or
                "รักษาได้ทุกโรค" in text.lower()
            ),
            "urgent_action": (
                "รีบ" in text.lower() or
                "ด่วน" in text.lower() or
                "ทันที" in text.lower()
            )
        }
        
        return risk_indicators

def evaluate_model():
    detector = FakeNewsDetector()
    results = []
    
    for item in fake_news_data:
        # Get prediction
        prediction = detector.predict(item['text'])
        
        # Analyze content
        risk_analysis = detector.analyze_content(item['text'])
        
        # Add to results
        results.append({
            **prediction,
            'expected_label': "ข่าวปลอม" if item['label'] == 1 else "ข่าวจริง",
            'is_correct': (prediction['prediction'] == "ข่าวปลอม") == (item['label'] == 1),
            'risk_indicators': risk_analysis
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการตรวจจับข่าวปลอม:\n")
    
    results = evaluate_model()
    
    # Print results grouped by correctness
    print("ผลการทำนายที่ถูกต้อง:")
    print("-" * 50)
    correct_results = [r for r in results if r['is_correct']]
    for result in correct_results:
        print(f"ข้อความ: {result['text']}")
        print(f"การทำนาย: {result['prediction']} (ความมั่นใจ: {result['confidence']:.4f})")
        print(f"ความน่าจะเป็นที่เป็นข่าวปลอม: {result['fake_probability']:.4f}")
        print(f"ตัวบ่งชี้ความเสี่ยง:")
        for indicator, present in result['risk_indicators'].items():
            if present:
                print(f"- {indicator}")
        print("-" * 50)
    
    print("\nผลการทำนายที่ไม่ถูกต้อง:")
    print("-" * 50)
    incorrect_results = [r for r in results if not r['is_correct']]
    for result in incorrect_results:
        print(f"ข้อความ: {result['text']}")
        print(f"การทำนาย: {result['prediction']} (ความมั่นใจ: {result['confidence']:.4f})")
        print(f"ค่าที่ถูกต้อง: {result['expected_label']}")
        print(f"ความน่าจะเป็นที่เป็นข่าวปลอม: {result['fake_probability']:.4f}")
        print("-" * 50)
    
    # Calculate metrics
    accuracy = len(correct_results) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    detector = FakeNewsDetector()
    
    print("\nทดสอบการตรวจจับข่าวปลอมแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความข่าวที่ต้องการตรวจสอบ (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        # Get prediction
        result = detector.predict(text)
        
        # Analyze content
        risk_analysis = detector.analyze_content(text)
        
        print("\nผลการวิเคราะห์:")
        print(f"การจัดประเภท: {result['prediction']}")
        print(f"ความมั่นใจ: {result['confidence']:.4f}")
        print(f"ความน่าจะเป็นที่เป็นข่าวปลอม: {result['fake_probability']:.4f}")
        
        if any(risk_analysis.values()):
            print("\nพบตัวบ่งชี้ที่น่าสงสัย:")
            for indicator, present in risk_analysis.items():
                if present:
                    print(f"- {indicator}")

def analyze_news_patterns():
    detector = FakeNewsDetector()
    
    print("\nการวิเคราะห์รูปแบบข่าวปลอม:")
    
    # Common fake news patterns
    patterns = [
        "ด่วน! พบวิธีรักษาโรคร้ายแรงด้วยสมุนไพรธรรมชาติ 100%",
        "แชร์ด่วน! รัฐบาลแจกเงิน 50,000 บาทให้ทุกคน",
        "เตือนภัย! อย่าเปิดไฟล์นี้เด็ดขาด มิฉะนั้นโทรศัพท์จะระเบิด",
        "นักวิทยาศาสตร์เผย! ดื่มน้ำเปล่าวันละ 10 ลิตรช่วยให้อายุยืน 150 ปี"
    ]
    
    for text in patterns:
        result = detector.predict(text)
        analysis = detector.analyze_content(text)
        
        print(f"\nข้อความ: {text}")
        print(f"การวิเคราะห์: {result['prediction']}")
        print(f"ความน่าจะเป็นที่เป็นข่าวปลอม: {result['fake_probability']:.4f}")
        print("ตัวบ่งชี้ที่พบ:")
        for indicator, present in analysis.items():
            if present:
                print(f"- {indicator}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลตรวจจับข่าวปลอม...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze common fake news patterns
    analyze_news_patterns()