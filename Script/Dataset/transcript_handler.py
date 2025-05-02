import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import pandas as pd
import os
import re
from Dataset.stt_transcript_dataset import stt_texts

class TranscriptHandler:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        
        # Define environment types and their characteristics
        self.environments = {
            'conversation': {'markers': ['A:', 'B:', 'สวัสดี', 'ครับ', 'ค่ะ']},
            'announcement': {'markers': ['ประกาศ', 'โปรดทราบ', 'ขอความร่วมมือ']},
            'news': {'markers': ['รายงานข่าว', 'ติดตามข่าว', 'ขณะนี้']},
            'classroom': {'markers': ['นักเรียน', 'บทเรียน', 'หน้า']},
            'hospital': {'markers': ['ผู้ป่วย', 'คุณหมอ', 'ห้องตรวจ']},
            'restaurant': {'markers': ['สั่งอาหาร', 'รับประทาน', 'จาน', 'แก้ว']},
            'transportation': {'markers': ['สถานี', 'เที่ยวบิน', 'ผู้โดยสาร']},
            'meeting': {'markers': ['ประชุม', 'โปรเจกต์', 'ข้อมูล', 'รายงาน']}
        }
    
    def identify_environment(self, text):
        # Clean the text
        cleaned_text = re.sub(r'\([^)]*\)', '', text).strip()
        
        # Count environment markers
        environment_scores = {}
        for env, properties in self.environments.items():
            score = sum(1 for marker in properties['markers'] if marker in cleaned_text)
            environment_scores[env] = score
        
        # Get environment with highest score
        detected_env = max(environment_scores.items(), key=lambda x: x[1])
        return {
            'environment': detected_env[0] if detected_env[1] > 0 else 'unknown',
            'confidence': detected_env[1] / len(self.environments[detected_env[0]]['markers'])
            if detected_env[1] > 0 else 0
        }
    
    def extract_speakers(self, text):
        # Find speaker turns using pattern A: text B: text
        speakers = {}
        turns = re.findall(r'([A-Z]):\s*([^A-Z:]+)(?=[A-Z]:|$)', text)
        
        for speaker, content in turns:
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(content.strip())
        
        return speakers
    
    def clean_transcript(self, text):
        # Remove environment annotations
        cleaned = re.sub(r'\([^)]*\)', '', text)
        # Remove speaker markers
        cleaned = re.sub(r'[A-Z]:\s*', '', cleaned)
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def format_dialogue(self, text):
        # Extract speakers and their turns
        speakers = self.extract_speakers(text)
        
        formatted = []
        for speaker, turns in speakers.items():
            for turn in turns:
                formatted.append(f"Speaker {speaker}: {turn}")
        
        return '\n'.join(formatted)
    
    def analyze_speech_patterns(self, text):
        # Clean text
        cleaned_text = self.clean_transcript(text)
        words = cleaned_text.split()
        
        # Calculate basic statistics
        stats = {
            'word_count': len(words),
            'avg_word_length': sum(len(word) for word in words) / len(words),
            'has_question': '?' in cleaned_text,
            'formality_level': self.estimate_formality(cleaned_text)
        }
        
        return stats
    
    def estimate_formality(self, text):
        formal_markers = ['ครับ', 'ค่ะ', 'ท่าน', 'กรุณา', 'โปรด']
        informal_markers = ['จ้า', 'นะ', 'อ่ะ', 'ดิ', 'เลย']
        
        formal_count = sum(1 for marker in formal_markers if marker in text)
        informal_count = sum(1 for marker in informal_markers if marker in text)
        
        if formal_count > informal_count:
            return 'formal'
        elif informal_count > formal_count:
            return 'informal'
        else:
            return 'neutral'

def analyze_transcripts():
    handler = TranscriptHandler()
    results = []
    
    for text in stt_texts:
        # Identify environment
        env = handler.identify_environment(text)
        
        # Extract speech patterns
        patterns = handler.analyze_speech_patterns(text)
        
        # Get speakers if it's a dialogue
        speakers = handler.extract_speakers(text) if 'A:' in text else {}
        
        results.append({
            'text': text,
            'environment': env['environment'],
            'environment_confidence': env['confidence'],
            'speech_patterns': patterns,
            'speakers': speakers,
            'cleaned_text': handler.clean_transcript(text)
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการวิเคราะห์บทสนทนา:\n")
    
    results = analyze_transcripts()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print("-" * 50)
        print(f"ข้อความต้นฉบับ: {result['text']}")
        print(f"\nสภาพแวดล้อม: {result['environment']} (ความมั่นใจ: {result['environment_confidence']:.2f})")
        
        if result['speakers']:
            print("\nผู้พูด:")
            for speaker, turns in result['speakers'].items():
                print(f"\nSpeaker {speaker}:")
                for turn in turns:
                    print(f"- {turn}")
        
        print("\nรูปแบบการพูด:")
        for key, value in result['speech_patterns'].items():
            print(f"- {key}: {value}")
        
        print(f"\nข้อความที่ทำความสะอาด: {result['cleaned_text']}")
        print("-" * 50)

def run_interactive_demo():
    handler = TranscriptHandler()
    
    print("\nทดสอบการวิเคราะห์บทสนทนาแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่บทสนทนาหรือข้อความ (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        # Analyze text
        env = handler.identify_environment(text)
        patterns = handler.analyze_speech_patterns(text)
        speakers = handler.extract_speakers(text)
        
        print("\nผลการวิเคราะห์:")
        print(f"สภาพแวดล้อม: {env['environment']} (ความมั่นใจ: {env['confidence']:.2f})")
        
        if speakers:
            print("\nผู้พูด:")
            for speaker, turns in speakers.items():
                print(f"\nSpeaker {speaker}:")
                for turn in turns:
                    print(f"- {turn}")
        
        print("\nรูปแบบการพูด:")
        for key, value in patterns.items():
            print(f"- {key}: {value}")

def analyze_environment_patterns():
    handler = TranscriptHandler()
    
    print("\nการวิเคราะห์รูปแบบสภาพแวดล้อมต่างๆ:")
    
    example_texts = {
        "การประชุม": "(เสียงในที่ประชุม) ตามที่ได้เสนอไปในวาระที่หนึ่ง มีท่านใดมีข้อซักถามไหมครับ",
        "ประกาศ": "(เสียงประกาศ) ขอเชิญผู้โดยสารทุกท่านขึ้นเครื่องได้ที่ประตู B5 ค่ะ",
        "ร้านอาหาร": "พี่ครับ ขอสั่งข้าวผัดกระเพราไก่ไข่ดาว เผ็ดน้อย จานนึงครับ",
        "บทสนทนา": "A: เป็นไงบ้าง งานเสร็จหรือยัง B: ใกล้เสร็จแล้ว อีกแป๊บเดียว"
    }
    
    for desc, text in example_texts.items():
        print(f"\nสถานการณ์: {desc}")
        print(f"ข้อความ: {text}")
        
        env = handler.identify_environment(text)
        patterns = handler.analyze_speech_patterns(text)
        
        print(f"สภาพแวดล้อมที่ตรวจพบ: {env['environment']}")
        print(f"ความมั่นใจ: {env['confidence']:.2f}")
        print("\nรูปแบบการพูด:")
        for key, value in patterns.items():
            print(f"- {key}: {value}")
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลวิเคราะห์บทสนทนา...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze environment patterns
    analyze_environment_patterns()