import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from Dataset.customer_support_ticket_classification_dataset import ticket_data

class CustomerSupportClassifier:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.get_categories())
        )
        self.model.eval()
        
        # Create label mappings
        self.categories = self.get_categories()
        self.label2id = {label: idx for idx, label in enumerate(self.categories)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Define category descriptions
        self.category_descriptions = {
            'shipping_issue': 'ปัญหาเกี่ยวกับการจัดส่งสินค้า',
            'payment_problem': 'ปัญหาเกี่ยวกับการชำระเงิน',
            'service_request': 'คำขอบริการต่างๆ',
            'product_inquiry': 'สอบถามข้อมูลสินค้า'
        }
    
    @staticmethod
    def get_categories():
        return list(ticket_data.keys())
    
    def classify_ticket(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get top prediction and confidence
        prediction = torch.argmax(probabilities, dim=1).item()
        
        # Get probabilities for all categories
        category_probs = {
            category: probabilities[0][self.label2id[category]].item()
            for category in self.categories
        }
        
        return {
            'text': text,
            'category': self.id2label[prediction],
            'confidence': probabilities[0][prediction].item(),
            'category_probabilities': category_probs
        }
    
    def analyze_keywords(self, text):
        # Define keywords for each category
        keywords = {
            'shipping_issue': ['จัดส่ง', 'พัสดุ', 'ขนส่ง', 'ได้รับ', 'ที่อยู่'],
            'payment_problem': ['ชำระเงิน', 'โอนเงิน', 'บัตร', 'ส่วนลด', 'คืนเงิน'],
            'service_request': ['สอบถาม', 'สมัคร', 'ขอ', 'แนะนำ', 'ติดต่อ'],
            'product_inquiry': ['สินค้า', 'ขนาด', 'สี', 'รับประกัน', 'คุณสมบัติ']
        }
        
        # Count keywords for each category
        keyword_counts = {}
        for category, words in keywords.items():
            count = sum(1 for word in words if word in text)
            keyword_counts[category] = count
        
        return keyword_counts

def evaluate_model():
    classifier = CustomerSupportClassifier()
    results = []
    
    # Process each category
    for category, texts in ticket_data.items():
        for text in texts:
            # Get prediction
            prediction = classifier.classify_ticket(text)
            
            # Analyze keywords
            keyword_analysis = classifier.analyze_keywords(text)
            
            results.append({
                'text': text,
                'expected_category': category,
                'predicted_category': prediction['category'],
                'confidence': prediction['confidence'],
                'is_correct': prediction['category'] == category,
                'keyword_analysis': keyword_analysis
            })
    
    return results

def demonstrate_usage():
    print("ทดสอบการจำแนกประเภทการติดต่อลูกค้า:\n")
    
    results = evaluate_model()
    
    # Group results by correctness
    correct = [r for r in results if r['is_correct']]
    incorrect = [r for r in results if not r['is_correct']]
    
    print("ผลการทำนายที่ถูกต้อง:")
    print("-" * 50)
    for result in correct[:5]:  # Show first 5 examples
        print(f"ข้อความ: {result['text']}")
        print(f"หมวดหมู่ที่ทำนาย: {result['predicted_category']} (ความมั่นใจ: {result['confidence']:.4f})")
        print("\nคำสำคัญที่พบ:")
        for category, count in result['keyword_analysis'].items():
            if count > 0:
                print(f"- {category}: {count} คำ")
        print("-" * 50)
    
    print("\nผลการทำนายที่ไม่ถูกต้อง:")
    print("-" * 50)
    for result in incorrect:
        print(f"ข้อความ: {result['text']}")
        print(f"หมวดหมู่ที่ทำนาย: {result['predicted_category']} (ความมั่นใจ: {result['confidence']:.4f})")
        print(f"หมวดหมู่ที่ถูกต้อง: {result['expected_category']}")
        print("\nคำสำคัญที่พบ:")
        for category, count in result['keyword_analysis'].items():
            if count > 0:
                print(f"- {category}: {count} คำ")
        print("-" * 50)
    
    # Calculate metrics
    accuracy = len(correct) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    classifier = CustomerSupportClassifier()
    
    print("\nทดสอบการจำแนกประเภทแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการจำแนก (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        # Get prediction
        prediction = classifier.classify_ticket(text)
        
        # Analyze keywords
        keyword_analysis = classifier.analyze_keywords(text)
        
        print("\nผลการวิเคราะห์:")
        print(f"หมวดหมู่: {prediction['category']}")
        print(f"ความมั่นใจ: {prediction['confidence']:.4f}")
        
        print("\nความน่าจะเป็นของแต่ละหมวดหมู่:")
        for category, prob in sorted(
            prediction['category_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"- {category}: {prob:.4f}")
        
        print("\nคำสำคัญที่พบ:")
        for category, count in keyword_analysis.items():
            if count > 0:
                print(f"- {category}: {count} คำ")

def analyze_ticket_patterns():
    classifier = CustomerSupportClassifier()
    
    print("\nการวิเคราะห์รูปแบบการติดต่อต่างๆ:")
    
    example_tickets = {
        "ติดตามพัสดุ": "สั่งสินค้าไปนานแล้ว ยังไม่ได้รับของเลย ขอเช็คสถานะการจัดส่งหน่อยค่ะ",
        "ปัญหาการชำระเงิน": "โอนเงินไปแล้ว แต่ระบบยังไม่อัพเดทสถานะการชำระเงิน",
        "สอบถามข้อมูล": "อยากทราบว่าสินค้ารุ่นนี้มีรับประกันกี่ปี และซ่อมได้ที่ไหนบ้าง",
        "ขอความช่วยเหลือ": "ลืมรหัสผ่านเข้าระบบ ต้องการรีเซ็ตรหัสผ่านใหม่"
    }
    
    for description, text in example_tickets.items():
        print(f"\nกรณี: {description}")
        print(f"ข้อความ: {text}")
        
        # Get prediction
        prediction = classifier.classify_ticket(text)
        
        # Analyze keywords
        keyword_analysis = classifier.analyze_keywords(text)
        
        print("\nผลการวิเคราะห์:")
        print(f"หมวดหมู่: {prediction['category']}")
        print(f"ความมั่นใจ: {prediction['confidence']:.4f}")
        
        print("\nคำสำคัญที่พบ:")
        for category, count in keyword_analysis.items():
            if count > 0:
                print(f"- {category}: {count} คำ")
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลจำแนกประเภทการติดต่อลูกค้า...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze ticket patterns
    analyze_ticket_patterns()