import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from Dataset.emotion_detection_dataset import emotion_data

class EmotionDetector:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(emotion_data.keys())
        )
        self.model.eval()
        
        # Create label mapping
        self.emotions = list(emotion_data.keys())
        self.label2id = {label: idx for idx, label in enumerate(self.emotions)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Create emotion descriptions in Thai
        self.emotion_descriptions = {
            "joy": "ความสุข ความดีใจ",
            "sadness": "ความเศร้า ความทุกข์",
            "anger": "ความโกรธ ความไม่พอใจ",
            "love": "ความรัก ความห่วงใย"
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
        
        # Get top prediction and confidence
        prediction = torch.argmax(probabilities, dim=1).item()
        
        # Get probabilities for all emotions
        emotion_probs = {
            emotion: probabilities[0][self.label2id[emotion]].item()
            for emotion in self.emotions
        }
        
        return {
            'text': text,
            'emotion': self.id2label[prediction],
            'confidence': probabilities[0][prediction].item(),
            'emotion_probabilities': emotion_probs
        }
    
    def batch_predict(self, texts):
        return [self.predict(text) for text in texts]

def evaluate_model():
    detector = EmotionDetector()
    results = []
    
    # Process each emotion category
    for emotion, texts in emotion_data.items():
        for text in texts:
            prediction = detector.predict(text)
            prediction['expected_emotion'] = emotion
            prediction['is_correct'] = prediction['emotion'] == emotion
            results.append(prediction)
    
    return results

def demonstrate_usage():
    print("ทดสอบการวิเคราะห์อารมณ์ความรู้สึก:\n")
    
    results = evaluate_model()
    
    # Print results by emotion category
    for emotion in emotion_data.keys():
        print(f"\nผลการทำนายสำหรับอารมณ์ {emotion}:")
        print("-" * 50)
        
        emotion_results = [r for r in results if r['expected_emotion'] == emotion]
        correct = [r for r in emotion_results if r['is_correct']]
        
        print(f"ความแม่นยำ: {len(correct)}/{len(emotion_results)} ({len(correct)/len(emotion_results):.2%})")
        
        # Print some examples
        print("\nตัวอย่างการทำนาย:")
        for result in emotion_results[:3]:  # Show first 3 examples
            print(f"\nข้อความ: {result['text']}")
            print(f"ทำนาย: {result['emotion']} (ความมั่นใจ: {result['confidence']:.4f})")
            print(f"ค่าที่ถูกต้อง: {result['expected_emotion']}")
    
    # Calculate overall accuracy
    accuracy = sum(1 for r in results if r['is_correct']) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    detector = EmotionDetector()
    
    print("\nทดสอบการวิเคราะห์อารมณ์แบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการวิเคราะห์ (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        result = detector.predict(text)
        
        print(f"\nผลการวิเคราะห์:")
        print(f"อารมณ์หลัก: {result['emotion']} (ความมั่นใจ: {result['confidence']:.4f})")
        print("\nความน่าจะเป็นของแต่ละอารมณ์:")
        for emotion, prob in result['emotion_probabilities'].items():
            print(f"{emotion}: {prob:.4f}")

def analyze_mixed_emotions():
    detector = EmotionDetector()
    
    print("\nการวิเคราะห์อารมณ์ผสม:")
    
    # ตัวอย่างประโยคที่มีอารมณ์ผสม
    mixed_examples = [
        "ดีใจที่ได้เจอ แต่เสียใจที่ต้องจาก",
        "โกรธมากที่โดนหลอก แต่ก็ยังรักอยู่ดี",
        "เศร้าใจที่ต้องเลิกกัน แต่ก็มีความสุขกับความทรงจำดีๆ",
        "รักเธอมาก แต่โกรธที่เธอไม่เข้าใจ",
    ]
    
    for text in mixed_examples:
        result = detector.predict(text)
        print(f"\nข้อความ: {text}")
        print("การวิเคราะห์อารมณ์:")
        for emotion, prob in sorted(
            result['emotion_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"- {emotion}: {prob:.4f}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลวิเคราะห์อารมณ์...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze mixed emotions
    analyze_mixed_emotions()