import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
from Dataset.conversation_simulation_dataset import conversation_data
import re

class ConversationSimulator:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        
        # Define conversation types
        self.conversation_types = {
            'customer_service': ['สั่ง', 'บริการ', 'สอบถาม', 'ขอ'],
            'casual_chat': ['สวัสดี', 'อากาศ', 'เป็นไง', 'สบาย'],
            'information': ['ที่ไหน', 'อย่างไร', 'เมื่อไร', 'ทำไม'],
            'appointment': ['นัด', 'จอง', 'วันที่', 'เวลา'],
            'game': ['คำถาม', 'เกม', 'คะแนน', 'ถูกต้อง']
        }
    
    def parse_dialogue(self, dialogue_text):
        # Split dialogue into turns
        turns = []
        for line in dialogue_text.split('\n'):
            if ':' in line:
                speaker, text = line.split(':', 1)
                turns.append({
                    'speaker': speaker.strip(),
                    'text': text.strip()
                })
        return turns
    
    def identify_conversation_type(self, dialogue_text):
        # Count keywords for each type
        type_scores = {ctype: 0 for ctype in self.conversation_types}
        
        for ctype, keywords in self.conversation_types.items():
            for keyword in keywords:
                if keyword in dialogue_text.lower():
                    type_scores[ctype] += 1
        
        # Return type with highest score
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def analyze_sentiment(self, text):
        # Simple rule-based sentiment analysis
        positive_words = ['ขอบคุณ', 'ดี', 'สบาย', 'ยินดี', 'เรียบร้อย']
        negative_words = ['ขอโทษ', 'เสียใจ', 'ไม่สะดวก', 'ปัญหา']
        
        sentiment = 'neutral'
        if any(word in text.lower() for word in positive_words):
            sentiment = 'positive'
        elif any(word in text.lower() for word in negative_words):
            sentiment = 'negative'
        
        return sentiment
    
    def generate_response(self, dialogue_history, max_length=50):
        # Prepare input by combining dialogue history
        input_text = "ตอบ: " + '\n'.join([f"{turn['speaker']}: {turn['text']}" for turn in dialogue_history])
        
        # Generate response
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            top_p=0.9
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def evaluate_dialogues():
    simulator = ConversationSimulator()
    results = []
    
    for dialogue in conversation_data:
        # Parse dialogue
        turns = simulator.parse_dialogue(dialogue)
        
        # Analyze conversation
        conv_type = simulator.identify_conversation_type(dialogue)
        
        # Analyze sentiment for each turn
        sentiments = [
            simulator.analyze_sentiment(turn['text'])
            for turn in turns
        ]
        
        # Generate next response
        next_response = simulator.generate_response(turns)
        
        results.append({
            'dialogue': dialogue,
            'turns': turns,
            'conversation_type': conv_type,
            'sentiments': sentiments,
            'generated_response': next_response
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการจำลองบทสนทนา:\n")
    
    results = evaluate_dialogues()
    
    for i, result in enumerate(results, 1):
        print(f"\nบทสนทนาที่ {i}:")
        print("-" * 50)
        print(f"ประเภทบทสนทนา: {result['conversation_type']}")
        print("\nการวิเคราะห์บทสนทนา:")
        for turn, sentiment in zip(result['turns'], result['sentiments']):
            print(f"{turn['speaker']}: {turn['text']}")
            print(f"ความรู้สึก: {sentiment}")
        
        print("\nการตอบสนองที่สร้าง:")
        print(f"Agent: {result['generated_response']}")
        print("-" * 50)

def run_interactive_demo():
    simulator = ConversationSimulator()
    
    print("\nทดสอบการสนทนาแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    dialogue_history = []
    while True:
        print("\nUser: ", end='')
        user_input = input()
        if user_input.lower() == 'exit':
            break
        
        # Add user input to history
        dialogue_history.append({
            'speaker': 'User',
            'text': user_input
        })
        
        # Generate response
        response = simulator.generate_response(dialogue_history)
        print(f"Agent: {response}")
        
        # Add agent response to history
        dialogue_history.append({
            'speaker': 'Agent',
            'text': response
        })
        
        # Analyze current turn
        sentiment = simulator.analyze_sentiment(user_input)
        conv_type = simulator.identify_conversation_type('\n'.join([
            f"{turn['speaker']}: {turn['text']}"
            for turn in dialogue_history
        ]))
        
        print(f"\nการวิเคราะห์: ประเภท={conv_type}, ความรู้สึก={sentiment}")

def analyze_conversation_patterns():
    simulator = ConversationSimulator()
    
    print("\nการวิเคราะห์รูปแบบการสนทนา:")
    
    example_dialogues = {
        "บริการลูกค้า": """
User: สวัสดีครับ ต้องการสอบถามเกี่ยวกับการสั่งสินค้าครับ
Agent: สวัสดีค่ะ ยินดีให้ข้อมูลค่ะ สอบถามเรื่องอะไรดีคะ
User: อยากทราบว่าสินค้าจัดส่งใช้เวลากี่วันครับ
""",
        "การนัดหมาย": """
User: ขอนัดพบแพทย์ครับ
Agent: ได้ค่ะ ไม่ทราบว่าต้องการนัดวันไหนคะ
User: วันจันทร์หน้าได้ไหมครับ
""",
        "สนทนาทั่วไป": """
User: สวัสดีครับ วันนี้อากาศดีนะ
Agent: สวัสดีค่ะ ใช่ค่ะ อากาศดีมากเลย
User: เหมาะกับการออกไปเที่ยวเลย
"""
    }
    
    for category, dialogue in example_dialogues.items():
        print(f"\nหมวด: {category}")
        turns = simulator.parse_dialogue(dialogue)
        
        print("การวิเคราะห์:")
        for turn in turns:
            sentiment = simulator.analyze_sentiment(turn['text'])
            print(f"{turn['speaker']}: {turn['text']}")
            print(f"ความรู้สึก: {sentiment}")
        
        print("\nการตอบสนองที่สร้าง:")
        response = simulator.generate_response(turns)
        print(f"Agent: {response}")
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลจำลองบทสนทนา...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze conversation patterns
    analyze_conversation_patterns()