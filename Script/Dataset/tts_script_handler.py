import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import re
from Dataset.tts_script_dataset import tts_texts

class TTSScriptHandler:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Define types of speech and their characteristics
        self.speech_types = {
            'greeting': ['สวัสดี', 'ยินดี', 'อรุณสวัสดิ์', 'ราตรีสวัสดิ์'],
            'announcement': ['ประกาศ', 'แจ้ง', 'โปรดทราบ'],
            'instruction': ['กรุณา', 'โปรด', 'ควร', 'ต้อง'],
            'emotional': ['ดีใจ', 'เสียใจ', 'โกรธ', 'ว้าว', 'กังวล'],
            'service': ['ขอบคุณ', 'ขออภัย', 'ยินดีให้บริการ'],
            'narrative': ['เล่า', 'นิทาน', 'รายงาน', 'สรุป']
        }
        
        # Define emotion markers
        self.emotion_markers = {
            'happy': ['ดีใจ', 'ยินดี', 'สุดยอด', 'ว้าว'],
            'sad': ['เสียใจ', 'เศร้า', 'กังวล'],
            'angry': ['โกรธ', 'ไม่พอใจ'],
            'neutral': ['กรุณา', 'โปรด', 'ขอบคุณ'],
            'excited': ['ว้าว', 'สุดยอด', 'เยี่ยม']
        }
    
    def analyze_script(self, text):
        # Identify speech type
        speech_type = self.identify_speech_type(text)
        
        # Identify emotion
        emotion = self.identify_emotion(text)
        
        # Analyze text properties
        properties = self.analyze_text_properties(text)
        
        return {
            'speech_type': speech_type,
            'emotion': emotion,
            'properties': properties
        }
    
    def identify_speech_type(self, text):
        type_scores = {}
        for stype, markers in self.speech_types.items():
            score = sum(1 for marker in markers if marker in text)
            type_scores[stype] = score
        
        # Get type with highest score
        speech_type = max(type_scores.items(), key=lambda x: x[1])
        return {
            'type': speech_type[0] if speech_type[1] > 0 else 'general',
            'confidence': speech_type[1] / len(self.speech_types[speech_type[0]])
            if speech_type[1] > 0 else 0
        }
    
    def identify_emotion(self, text):
        emotion_scores = {}
        for emotion, markers in self.emotion_markers.items():
            score = sum(1 for marker in markers if marker in text)
            emotion_scores[emotion] = score
        
        # Get emotion with highest score
        emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return {
            'emotion': emotion[0] if emotion[1] > 0 else 'neutral',
            'intensity': emotion[1] / len(self.emotion_markers[emotion[0]])
            if emotion[1] > 0 else 0
        }
    
    def analyze_text_properties(self, text):
        # Count punctuation marks
        exclamations = text.count('!')
        questions = text.count('?')
        
        # Check for special characters or markers
        has_parentheses = bool(re.search(r'\(.*?\)', text))
        has_ellipsis = '...' in text
        
        # Analyze sentence structure
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        return {
            'exclamations': exclamations,
            'questions': questions,
            'has_parentheses': has_parentheses,
            'has_ellipsis': has_ellipsis,
            'word_count': len(words),
            'avg_word_length': avg_word_length
        }
    
    def prepare_for_tts(self, text):
        # Clean text for TTS
        cleaned = text.strip()
        
        # Remove parenthetical annotations
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        
        # Normalize punctuation
        cleaned = re.sub(r'!+', '!', cleaned)  # Multiple ! to single !
        cleaned = re.sub(r'\?+', '?', cleaned)  # Multiple ? to single ?
        cleaned = re.sub(r'\.{2,}', '...', cleaned)  # Normalize ellipsis
        
        return cleaned
    
    def suggest_speech_parameters(self, text):
        analysis = self.analyze_script(text)
        
        # Suggest parameters based on analysis
        suggestions = {
            'rate': 'normal',
            'pitch': 'normal',
            'volume': 'normal',
            'style': 'neutral'
        }
        
        # Adjust based on emotion
        if analysis['emotion']['emotion'] == 'happy':
            suggestions['rate'] = 'slightly_fast'
            suggestions['pitch'] = 'high'
            suggestions['style'] = 'cheerful'
        elif analysis['emotion']['emotion'] == 'sad':
            suggestions['rate'] = 'slow'
            suggestions['pitch'] = 'low'
            suggestions['style'] = 'gentle'
        elif analysis['emotion']['emotion'] == 'angry':
            suggestions['rate'] = 'fast'
            suggestions['pitch'] = 'high'
            suggestions['volume'] = 'loud'
            suggestions['style'] = 'strong'
        
        # Adjust based on speech type
        if analysis['speech_type']['type'] == 'announcement':
            suggestions['volume'] = 'loud'
            suggestions['style'] = 'formal'
        elif analysis['speech_type']['type'] == 'instruction':
            suggestions['rate'] = 'slow'
            suggestions['style'] = 'clear'
        
        return suggestions

def evaluate_scripts():
    handler = TTSScriptHandler()
    results = []
    
    for text in tts_texts:
        # Analyze script
        analysis = handler.analyze_script(text)
        
        # Get TTS preparation
        prepared_text = handler.prepare_for_tts(text)
        
        # Get speech parameter suggestions
        suggestions = handler.suggest_speech_parameters(text)
        
        results.append({
            'original_text': text,
            'prepared_text': prepared_text,
            'analysis': analysis,
            'suggestions': suggestions
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการวิเคราะห์สคริปต์ TTS:\n")
    
    results = evaluate_scripts()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print("-" * 50)
        print(f"ข้อความต้นฉบับ: {result['original_text']}")
        print(f"ข้อความที่เตรียมสำหรับ TTS: {result['prepared_text']}")
        
        print("\nการวิเคราะห์:")
        print(f"ประเภทการพูด: {result['analysis']['speech_type']['type']}")
        print(f"อารมณ์: {result['analysis']['emotion']['emotion']}")
        print(f"ความเข้มข้นของอารมณ์: {result['analysis']['emotion']['intensity']:.2f}")
        
        print("\nคุณสมบัติข้อความ:")
        for key, value in result['analysis']['properties'].items():
            print(f"- {key}: {value}")
        
        print("\nคำแนะนำสำหรับการสังเคราะห์เสียง:")
        for param, value in result['suggestions'].items():
            print(f"- {param}: {value}")
        print("-" * 50)

def run_interactive_demo():
    handler = TTSScriptHandler()
    
    print("\nทดสอบการวิเคราะห์สคริปต์แบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการวิเคราะห์ (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        # Analyze text
        analysis = handler.analyze_script(text)
        prepared = handler.prepare_for_tts(text)
        suggestions = handler.suggest_speech_parameters(text)
        
        print("\nผลการวิเคราะห์:")
        print(f"ข้อความที่เตรียมสำหรับ TTS: {prepared}")
        print(f"\nประเภทการพูด: {analysis['speech_type']['type']}")
        print(f"อารมณ์: {analysis['emotion']['emotion']}")
        
        print("\nคุณสมบัติข้อความ:")
        for key, value in analysis['properties'].items():
            print(f"- {key}: {value}")
        
        print("\nคำแนะนำสำหรับการสังเคราะห์เสียง:")
        for param, value in suggestions.items():
            print(f"- {param}: {value}")

def analyze_script_patterns():
    handler = TTSScriptHandler()
    
    print("\nการวิเคราะห์รูปแบบสคริปต์ต่างๆ:")
    
    example_scripts = {
        "ทักทาย": "สวัสดีครับ ยินดีต้อนรับทุกท่านครับ!",
        "ประกาศ": "ขอความสนใจทุกท่าน มีประกาศสำคัญ...",
        "อารมณ์": "ว้าว! ดีใจมากเลยที่ได้รับรางวัลนี้!",
        "คำแนะนำ": "กรุณาตรวจสอบสัมภาระก่อนออกจากรถ"
    }
    
    for desc, text in example_scripts.items():
        print(f"\nรูปแบบ: {desc}")
        print(f"ข้อความ: {text}")
        
        analysis = handler.analyze_script(text)
        suggestions = handler.suggest_speech_parameters(text)
        
        print("\nการวิเคราะห์:")
        print(f"ประเภทการพูด: {analysis['speech_type']['type']}")
        print(f"อารมณ์: {analysis['emotion']['emotion']}")
        
        print("\nคำแนะนำสำหรับการสังเคราะห์เสียง:")
        for param, value in suggestions.items():
            print(f"- {param}: {value}")
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลวิเคราะห์สคริปต์ TTS...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze script patterns
    analyze_script_patterns()