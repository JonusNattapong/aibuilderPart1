import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import re
import csv
import numpy as np

class UncensoredContentModerator:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        """
        ตัวตรวจสอบเนื้อหาที่ไม่เหมาะสมหลายประเภท
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
        self.model.eval()
        
        # กำหนดประเภทของเนื้อหาที่ไม่เหมาะสม
        self.id2label = {
            0: "ปกติ",  # normal content
            1: "คำหยาบคาย",  # offensive language
            2: "เนื้อหาลามก",  # adult content
            3: "การเกลียดชัง",  # hate speech
            4: "ข้อมูลเท็จ",  # misinformation
            5: "กิจกรรมผิดกฎหมาย",  # illegal activities
            6: "ความรุนแรง",  # violence
            7: "การทำร้ายตัวเอง"  # self-harm
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # นิยามคำต้องห้ามและรูปแบบของเนื้อหาที่ไม่เหมาะสม
        self.patterns = {
            "offensive_patterns": [
                r'ไอ้[ก-ฮ]+', r'อี[ก-ฮ]+', r'มึง', r'กู', r'เหี้ย', r'หน้าด้าน',
                r'ควาย', r'สัตว์', r'โง่', r'บ้า', r'ไอ้[ก-ฮ]+'
            ],
            "adult_patterns": [
                r'เซ็กส์', r'โป๊', r'ลามก', r'อนาจาร', r'เย็ด', r'นม',
                r'เปลือย', r'ค(ว|รรว)ย', r'จู๋', r'สำเร็จความใคร่', r'เสียว'
            ],
            "hate_patterns": [
                r'ไอ้พวก[ก-ฮ]+', r'อี[ก-ฮ]+', r'กระเทย', r'ตุ๊ด', r'แขก', r'จีน',
                r'ฝรั่ง', r'ต่างด้าว', r'นิโกร', r'ยิว', r'เกลียด[ก-ฮ]+'
            ],
            "misinfo_patterns": [
                r'แชร์ด่วน', r'100%', r'ห้ามแชร์', r'รักษาได้ทันที', r'รับรอง',
                r'กินแล้วหาย', r'ลดน้ำหนักได้', r'หายขาด', r'พลังจักรวาล'
            ],
            "illegal_patterns": [
                r'ยาเสพติด', r'กัญชา', r'โคเคน', r'ปืน', r'ระเบิด', r'ฟอกเงิน',
                r'ปลอม', r'ขโมย', r'ลักทรัพย์', r'เจาะระบบ', r'แฮก'
            ],
            "violence_patterns": [
                r'ฆ่า', r'ตาย', r'แทง', r'เชือด', r'ระเบิด', r'ยิง', r'ทรมาน',
                r'ทำร้าย', r'เลือด', r'ฆาตกรรม', r'ฆ่าตัวตาย'
            ],
            "self_harm_patterns": [
                r'ฆ่าตัวตาย', r'กรีดข้อมือ', r'กินยา[ก-ฮ]+เยอะๆ', r'กระโดดตึก', r'แขวนคอ',
                r'หมดหวัง', r'ไม่อยากมีชีวิตอยู่', r'จบชีวิต', r'จบทุกอย่าง'
            ]
        }
    
    def predict(self, text):
        """ทำนายประเภทของเนื้อหา"""
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # หาค่าความน่าจะเป็นสูงสุด
        prediction = torch.argmax(probabilities, dim=1).item()
        
        # คำนวณค่าความเชื่อมั่นของทุกประเภท
        all_probs = {self.id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        
        # ตรวจจับรูปแบบเนื้อหาที่ไม่เหมาะสมด้วย regex
        detected_patterns = {}
        for pattern_type, pattern_list in self.patterns.items():
            detected = []
            for pattern in pattern_list:
                matches = re.findall(pattern, text.lower())
                if matches:
                    detected.extend(matches)
            if detected:
                detected_patterns[pattern_type] = list(set(detected))
        
        return {
            'text': text,
            'prediction': self.id2label[prediction],
            'confidence': probabilities[0][prediction].item(),
            'all_probabilities': all_probs,
            'detected_patterns': detected_patterns,
            'is_inappropriate': prediction != 0
        }
    
    def batch_predict(self, texts):
        """ทำนายหลายข้อความพร้อมกัน"""
        return [self.predict(text) for text in texts]
    
    def analyze_content(self, text):
        """วิเคราะห์เนื้อหาเชิงลึก"""
        prediction = self.predict(text)
        
        # คำนวณความเสี่ยงโดยรวม (0-100%)
        risk_score = 0
        
        if prediction['is_inappropriate']:
            risk_score += 50  # มีความเสี่ยงพื้นฐานสูงถ้าโมเดลทำนายว่าไม่เหมาะสม
        
        patterns_found = len(prediction['detected_patterns'])
        if patterns_found > 0:
            risk_score += min(patterns_found * 10, 30)  # เพิ่มคะแนนความเสี่ยงตามรูปแบบที่พบ แต่ไม่เกิน 30%
        
        # เพิ่มคะแนนความเสี่ยงตามความมั่นใจของโมเดล
        if prediction['confidence'] > 0.8:
            risk_score += 20
        elif prediction['confidence'] > 0.6:
            risk_score += 10
        
        # ปรับคะแนนขั้นสุดท้าย
        risk_score = min(risk_score, 100)
        
        return {
            'content_type': prediction['prediction'],
            'risk_score': risk_score,
            'risk_level': 'สูงมาก' if risk_score > 75 else 'สูง' if risk_score > 50 else 'ปานกลาง' if risk_score > 25 else 'ต่ำ',
            'detected_patterns': prediction['detected_patterns'],
            'recommend_action': 'บล็อก' if risk_score > 75 else 'ตรวจสอบเพิ่มเติม' if risk_score > 25 else 'อนุญาต'
        }

def load_uncensored_dataset():
    """โหลดชุดข้อมูล uncensored"""
    dataset_path = os.path.join('DataOutput', 'thai_uncensored_dataset.csv')
    if not os.path.exists(dataset_path):
        print(f"ไม่พบไฟล์ {dataset_path} กรุณารันสคริปต์ DatasetCook/uncensored_dataset.py ก่อน")
        return []
    
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # ข้าม header
        for row in reader:
            if len(row) >= 3:
                data.append({
                    'id': row[0],
                    'text': row[1],
                    'category': row[2]
                })
    return data

def evaluate_model():
    """ประเมินโมเดลกับชุดข้อมูล uncensored"""
    dataset = load_uncensored_dataset()
    if not dataset:
        return []
    
    moderator = UncensoredContentModerator()
    results = []
    
    for item in dataset:
        # ทำนาย
        prediction = moderator.predict(item['text'])
        
        # วิเคราะห์
        analysis = moderator.analyze_content(item['text'])
        
        # ตรวจสอบว่าทำนายถูกต้องหรือไม่ (ถ้าไม่ใช่ "ปกติ" ถือว่าเป็นเนื้อหาไม่เหมาะสม)
        expected_inappropriate = item['category'] != "normal"
        is_correct = prediction['is_inappropriate'] == expected_inappropriate
        
        # เพิ่มผลลัพธ์
        results.append({
            'text': item['text'],
            'category': item['category'],
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'risk_score': analysis['risk_score'],
            'risk_level': analysis['risk_level'],
            'detected_patterns': prediction['detected_patterns'],
            'recommend_action': analysis['recommend_action'],
            'is_correct': is_correct
        })
    
    return results

def demonstrate_usage():
    """สาธิตการใช้งานโมเดล"""
    print("ทดสอบการตรวจสอบเนื้อหาที่ไม่เหมาะสม:\n")
    
    results = evaluate_model()
    if not results:
        print("ไม่มีข้อมูลสำหรับทดสอบ กรุณารันสคริปต์สร้างข้อมูลก่อน")
        return
    
    # แสดงผลแยกตามหมวดหมู่
    categories = set(r['category'] for r in results)
    for category in categories:
        category_results = [r for r in results if r['category'] == category]
        correct = [r for r in category_results if r['is_correct']]
        
        print(f"\nประเภท: {category}")
        print("-" * 50)
        print(f"ความแม่นยำ: {len(correct)}/{len(category_results)} ({len(correct)/len(category_results):.2%})")
        
        # แสดงตัวอย่าง
        if category_results:
            sample = category_results[0]
            print(f"\nตัวอย่าง:")
            print(f"ข้อความ: {sample['text']}")
            print(f"ทำนาย: {sample['prediction']} (ความมั่นใจ: {sample['confidence']:.2f})")
            print(f"คะแนนความเสี่ยง: {sample['risk_score']} - {sample['risk_level']}")
            if sample['detected_patterns']:
                print("รูปแบบที่ตรวจพบ:", sample['detected_patterns'])
            print(f"คำแนะนำ: {sample['recommend_action']}")
    
    # คำนวณความแม่นยำโดยรวม
    accuracy = len([r for r in results if r['is_correct']]) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    """เรียกใช้โหมดโต้ตอบเพื่อทดสอบการตรวจสอบเนื้อหา"""
    print("\n=== โหมดโต้ตอบ: ตรวจสอบเนื้อหาที่ไม่เหมาะสม ===")
    print("พิมพ์ข้อความเพื่อตรวจสอบ (พิมพ์ 'exit' เพื่อออก)")
    
    moderator = UncensoredContentModerator()
    
    while True:
        user_input = input("\nป้อนข้อความ: ")
        if user_input.lower() in ['exit', 'quit', 'q', 'ออก']:
            break
        
        prediction = moderator.predict(user_input)
        analysis = moderator.analyze_content(user_input)
        
        print(f"\nผลการวิเคราะห์:")
        print(f"ประเภทเนื้อหา: {prediction['prediction']} (ความมั่นใจ: {prediction['confidence']:.2f})")
        print(f"คะแนนความเสี่ยง: {analysis['risk_score']} - {analysis['risk_level']}")
        
        if prediction['detected_patterns']:
            print("\nรูปแบบที่ตรวจพบ:")
            for pattern_type, patterns in prediction['detected_patterns'].items():
                print(f"- {pattern_type}: {', '.join(patterns)}")
        
        print(f"\nคำแนะนำ: {analysis['recommend_action']}")

def analyze_moderation_effectiveness():
    """วิเคราะห์ประสิทธิภาพของโมเดลในการกรองเนื้อหาแต่ละประเภท"""
    results = evaluate_model()
    if not results:
        print("ไม่มีข้อมูลสำหรับวิเคราะห์ กรุณารันสคริปต์สร้างข้อมูลก่อน")
        return
    
    print("\n=== การวิเคราะห์ประสิทธิภาพของโมเดลกรองเนื้อหา ===")
    
    # วิเคราะห์แยกตามประเภทเนื้อหา
    categories = set(r['category'] for r in results)
    
    # สร้างตารางสรุป
    print("\nประสิทธิภาพแยกตามประเภทเนื้อหา:")
    print("-" * 80)
    print(f"{'ประเภทเนื้อหา':<25} {'ความแม่นยำ':<15} {'เฉลี่ยคะแนนความเสี่ยง':<20} {'อัตราการตรวจพบ':<15}")
    print("-" * 80)
    
    for category in categories:
        category_results = [r for r in results if r['category'] == category]
        correct = [r for r in category_results if r['is_correct']]
        accuracy = len(correct)/len(category_results)
        
        avg_risk = sum(r['risk_score'] for r in category_results) / len(category_results)
        
        detection_rate = len([r for r in category_results if r['detected_patterns']]) / len(category_results)
        
        print(f"{category:<25} {accuracy:.2%:<15} {avg_risk:.2f}{'%':<18} {detection_rate:.2%:<15}")
    
    # วิเคราะห์คะแนนความเสี่ยงโดยรวม
    risk_scores = [r['risk_score'] for r in results]
    print(f"\nสถิติคะแนนความเสี่ยงโดยรวม:")
    print(f"- ค่าเฉลี่ย: {sum(risk_scores)/len(risk_scores):.2f}%")
    print(f"- ค่าต่ำสุด: {min(risk_scores):.2f}%")
    print(f"- ค่าสูงสุด: {max(risk_scores):.2f}%")
    
    # แสดงเนื้อหาที่ตรวจจับยากที่สุด
    hard_to_detect = [r for r in results if not r['is_correct']]
    if hard_to_detect:
        print("\nตัวอย่างเนื้อหาที่ตรวจจับได้ยาก:")
        for i, result in enumerate(hard_to_detect[:3], 1):
            print(f"\n{i}. ข้อความ: {result['text']}")
            print(f"   ประเภทจริง: {result['category']}")
            print(f"   ทำนายเป็น: {result['prediction']} (ความมั่นใจ: {result['confidence']:.2f})")
            print(f"   คะแนนความเสี่ยง: {result['risk_score']}% - {result['risk_level']}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลตรวจสอบเนื้อหาที่ไม่เหมาะสม...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if not os.path.exists(os.path.join('DataOutput', 'thai_uncensored_dataset.csv')):
        print("ไม่พบชุดข้อมูล thai_uncensored_dataset.csv")
        print("กรุณารันสคริปต์ DatasetCook/uncensored_dataset.py ก่อน")
        exit()
    
    # แสดงข้อความคำเตือน
    print("\nคำเตือน: สคริปต์นี้ใช้สำหรับวัตถุประสงค์ในการวิจัยระบบกรองเนื้อหาเท่านั้น")
    print("เนื้อหาที่ใช้ในการทดสอบอาจมีลักษณะไม่เหมาะสม ใช้เพื่อตรวจสอบประสิทธิภาพของระบบกรองเนื้อหา\n")
    
    # รันการสาธิต
    demonstrate_usage()
    
    # รันโหมดโต้ตอบ
    run_interactive_demo()
    
    # วิเคราะห์ประสิทธิภาพ
    analyze_moderation_effectiveness()
