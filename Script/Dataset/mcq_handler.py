import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForSequenceClassification
import pandas as pd
import os
import numpy as np
from Dataset.mcq_dataset import mcq_data

class MCQHandler:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMultipleChoice.from_pretrained(model_name)
        self.model.eval()
    
    def prepare_inputs(self, context, question, choices):
        # Prepare input for each choice
        encodings = []
        for choice in choices:
            # Format: [CLS] Context [SEP] Question + Choice [SEP]
            text = f"{context} [SEP] {question} [SEP] {choice}"
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            encodings.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })
        
        # Stack all choices
        input_ids = torch.cat([e['input_ids'] for e in encodings])
        attention_mask = torch.cat([e['attention_mask'] for e in encodings])
        
        return {
            'input_ids': input_ids.unsqueeze(0),
            'attention_mask': attention_mask.unsqueeze(0)
        }
    
    def predict(self, context, question, choices):
        # Prepare model inputs
        inputs = self.prepare_inputs(context, question, choices)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predictions and confidence scores
        predicted_idx = torch.argmax(probabilities).item()
        confidence_scores = probabilities[0].tolist()
        
        return {
            'predicted_answer': choices[predicted_idx],
            'confidence': confidence_scores[predicted_idx],
            'all_scores': {
                choice: score for choice, score in zip(choices, confidence_scores)
            }
        }
    
    def analyze_question(self, context, question):
        # Analyze question type and difficulty
        question_types = {
            'factual': ['ใคร', 'อะไร', 'ที่ไหน', 'เมื่อใด', 'กี่'],
            'analytical': ['อย่างไร', 'ทำไม', 'เพราะเหตุใด'],
            'comparative': ['เปรียบเทียบ', 'ต่างกัน', 'เหมือนกัน']
        }
        
        question_type = 'other'
        for qtype, keywords in question_types.items():
            if any(keyword in question for keyword in keywords):
                question_type = qtype
                break
        
        # Estimate difficulty based on question length and complexity
        difficulty = 'medium'
        if len(question.split()) < 5:
            difficulty = 'easy'
        elif len(question.split()) > 15 or question_type == 'analytical':
            difficulty = 'hard'
        
        return {
            'question_type': question_type,
            'difficulty': difficulty,
            'word_count': len(question.split()),
            'requires_context': bool(context)
        }

def evaluate_model():
    handler = MCQHandler()
    results = []
    
    for item in mcq_data:
        # Make prediction
        prediction = handler.predict(
            item['context'],
            item['question'],
            item['choices']
        )
        
        # Analyze question
        analysis = handler.analyze_question(
            item['context'],
            item['question']
        )
        
        # Add to results
        results.append({
            'context': item['context'],
            'question': item['question'],
            'choices': item['choices'],
            'correct_answer': item['answer_label'],
            'predicted_answer': prediction['predicted_answer'],
            'confidence': prediction['confidence'],
            'is_correct': prediction['predicted_answer'] == item['answer_label'],
            'analysis': analysis,
            'choice_scores': prediction['all_scores']
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการตอบคำถามแบบหลายตัวเลือก:\n")
    
    results = evaluate_model()
    
    # Group results by correctness
    correct = [r for r in results if r['is_correct']]
    incorrect = [r for r in results if not r['is_correct']]
    
    print("ผลการทำนายที่ถูกต้อง:")
    print("-" * 50)
    for result in correct:
        print(f"บริบท: {result['context']}")
        print(f"คำถาม: {result['question']}")
        print(f"ตัวเลือก: {', '.join(result['choices'])}")
        print(f"คำตอบที่ทำนาย: {result['predicted_answer']} (ความมั่นใจ: {result['confidence']:.4f})")
        print("\nการวิเคราะห์คำถาม:")
        for key, value in result['analysis'].items():
            print(f"- {key}: {value}")
        print("-" * 50)
    
    print("\nผลการทำนายที่ไม่ถูกต้อง:")
    print("-" * 50)
    for result in incorrect:
        print(f"บริบท: {result['context']}")
        print(f"คำถาม: {result['question']}")
        print(f"ตัวเลือก: {', '.join(result['choices'])}")
        print(f"คำตอบที่ถูกต้อง: {result['correct_answer']}")
        print(f"คำตอบที่ทำนาย: {result['predicted_answer']} (ความมั่นใจ: {result['confidence']:.4f})")
        print("\nคะแนนแต่ละตัวเลือก:")
        for choice, score in result['choice_scores'].items():
            print(f"- {choice}: {score:.4f}")
        print("-" * 50)
    
    # Calculate accuracy
    accuracy = len(correct) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    handler = MCQHandler()
    
    print("\nทดสอบการตอบคำถามแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่บริบท (หรือพิมพ์ 'exit' เพื่อออก):")
        context = input()
        if context.lower() == 'exit':
            break
        
        print("ใส่คำถาม:")
        question = input()
        
        print("ใส่ตัวเลือก (คั่นด้วยเครื่องหมาย ,):")
        choices_input = input()
        choices = [c.strip() for c in choices_input.split(',')]
        
        if len(choices) < 2:
            print("กรุณาใส่อย่างน้อย 2 ตัวเลือก")
            continue
        
        # Make prediction
        prediction = handler.predict(context, question, choices)
        
        # Analyze question
        analysis = handler.analyze_question(context, question)
        
        print("\nผลการทำนาย:")
        print(f"คำตอบที่เลือก: {prediction['predicted_answer']}")
        print(f"ความมั่นใจ: {prediction['confidence']:.4f}")
        
        print("\nคะแนนแต่ละตัวเลือก:")
        for choice, score in prediction['all_scores'].items():
            print(f"- {choice}: {score:.4f}")
        
        print("\nการวิเคราะห์คำถาม:")
        for key, value in analysis.items():
            print(f"- {key}: {value}")

def analyze_question_types():
    handler = MCQHandler()
    
    print("\nการวิเคราะห์ประเภทคำถามต่างๆ:")
    
    example_questions = {
        "ข้อเท็จจริง": {
            "context": "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย มีประชากรประมาณ 10 ล้านคน",
            "question": "เมืองหลวงของประเทศไทยคือที่ใด?",
            "choices": ["กรุงเทพมหานคร", "เชียงใหม่", "ภูเก็ต", "หาดใหญ่"]
        },
        "วิเคราะห์": {
            "context": "ปัญหาฝุ่น PM2.5 เกิดจากหลายสาเหตุ ทั้งการจราจร การเผาในที่โล่ง และโรงงานอุตสาหกรรม",
            "question": "เพราะเหตุใดปัญหาฝุ่น PM2.5 จึงแก้ไขได้ยาก?",
            "choices": [
                "มีสาเหตุหลายประการ",
                "ขาดเทคโนโลยี",
                "ประชาชนไม่ร่วมมือ",
                "งบประมาณไม่เพียงพอ"
            ]
        },
        "เปรียบเทียบ": {
            "context": "รถยนต์ไฟฟ้าใช้พลังงานสะอาด แต่มีราคาสูง ส่วนรถยนต์น้ำมันราคาถูกกว่าแต่ก่อมลพิษ",
            "question": "รถยนต์ไฟฟ้าและรถยนต์น้ำมันต่างกันอย่างไร?",
            "choices": [
                "ใช้พลังงานต่างชนิดกัน",
                "ราคาต่างกัน",
                "ผลกระทบต่อสิ่งแวดล้อม",
                "ถูกทุกข้อ"
            ]
        }
    }
    
    for qtype, data in example_questions.items():
        print(f"\nประเภท: {qtype}")
        print(f"บริบท: {data['context']}")
        print(f"คำถาม: {data['question']}")
        
        # Analyze question
        analysis = handler.analyze_question(data['context'], data['question'])
        print("\nผลการวิเคราะห์:")
        for key, value in analysis.items():
            print(f"- {key}: {value}")
        
        # Make prediction
        prediction = handler.predict(data['context'], data['question'], data['choices'])
        print(f"\nคำตอบที่เลือก: {prediction['predicted_answer']}")
        print(f"ความมั่นใจ: {prediction['confidence']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลตอบคำถามแบบหลายตัวเลือก...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze different question types
    analyze_question_types()