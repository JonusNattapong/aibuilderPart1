import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import os
import json
from Dataset.ner_dataset import ner_data

class NERHandler:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.get_entity_labels())
        )
        self.model.eval()
        
        # Define entity labels
        self.label2id = {
            'O': 0,
            'B-PERSON': 1,
            'I-PERSON': 2,
            'B-ORGANIZATION': 3,
            'I-ORGANIZATION': 4,
            'B-LOCATION': 5,
            'I-LOCATION': 6,
            'B-DATE': 7,
            'I-DATE': 8,
            'B-PERCENT': 9,
            'I-PERCENT': 10,
            'B-EVENT': 11,
            'I-EVENT': 12
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def get_entity_labels(self):
        return [
            'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE',
            'PERCENT', 'EVENT'
        ]
    
    def identify_entities(self, text):
        # Tokenize text
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in tokens.items() if k != "offset_mapping"})
            predictions = torch.argmax(outputs.logits, dim=2)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=2)
        
        # Extract entities
        entities = []
        current_entity = None
        offset_mapping = tokens.offset_mapping[0]
        
        for idx, (prediction, offset) in enumerate(zip(predictions[0], offset_mapping)):
            label = self.id2label[prediction.item()]
            confidence = probabilities[0][idx][prediction].item()
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': text[offset[0].item():offset[1].item()],
                    'type': label[2:],
                    'start': offset[0].item(),
                    'end': offset[1].item(),
                    'confidence': confidence
                }
            elif label.startswith('I-') and current_entity:
                if label[2:] == current_entity['type']:
                    current_entity['text'] += text[offset[0].item():offset[1].item()]
                    current_entity['end'] = offset[1].item()
                    current_entity['confidence'] = (current_entity['confidence'] + confidence) / 2
            elif label == 'O' and current_entity:
                entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def format_entities_for_display(self, text, entities):
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        # Format text with entity markup
        formatted = text
        offset = 0
        for entity in sorted_entities:
            start = entity['start'] + offset
            end = entity['end'] + offset
            markup = f"[{entity['text']}]({entity['type']})"
            formatted = formatted[:start] + markup + formatted[end:]
            offset += len(markup) - (end - start)
        
        return formatted

def evaluate_model():
    handler = NERHandler()
    results = []
    
    for item in ner_data:
        # Get predictions
        entities = handler.identify_entities(item['sentence'])
        
        # Format entities as expected in dataset
        expected_entities = [(e[0], e[1], e[2], e[3]) for e in item['entities']]
        predicted_entities = [
            (e['text'], e['type'], e['start'], e['end'])
            for e in entities
        ]
        
        # Add to results
        results.append({
            'sentence': item['sentence'],
            'predicted_entities': entities,
            'expected_entities': expected_entities,
            'formatted_text': handler.format_entities_for_display(item['sentence'], entities),
            'is_correct': set(predicted_entities) == set(expected_entities)
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการระบุชื่อเฉพาะ (NER):\n")
    
    results = evaluate_model()
    
    # Group results by correctness
    correct = [r for r in results if r['is_correct']]
    incorrect = [r for r in results if not r['is_correct']]
    
    print("ผลการทำนายที่ถูกต้อง:")
    print("-" * 50)
    for result in correct:
        print(f"ประโยค: {result['sentence']}")
        print("ชื่อเฉพาะที่พบ:")
        for entity in result['predicted_entities']:
            print(f"- {entity['text']} ({entity['type']}) [ความมั่นใจ: {entity['confidence']:.4f}]")
        print(f"ข้อความที่ทำเครื่องหมาย: {result['formatted_text']}")
        print("-" * 50)
    
    print("\nผลการทำนายที่ไม่ถูกต้อง:")
    print("-" * 50)
    for result in incorrect:
        print(f"ประโยค: {result['sentence']}")
        print("\nชื่อเฉพาะที่ทำนาย:")
        for entity in result['predicted_entities']:
            print(f"- {entity['text']} ({entity['type']}) [ความมั่นใจ: {entity['confidence']:.4f}]")
        print("\nชื่อเฉพาะที่ถูกต้อง:")
        for entity in result['expected_entities']:
            print(f"- {entity[0]} ({entity[1]})")
        print("-" * 50)
    
    # Calculate accuracy
    accuracy = len(correct) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    handler = NERHandler()
    
    print("\nทดสอบการระบุชื่อเฉพาะแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการวิเคราะห์ (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        # Identify entities
        entities = handler.identify_entities(text)
        
        print("\nชื่อเฉพาะที่พบ:")
        for entity in entities:
            print(f"- {entity['text']} ({entity['type']}) [ความมั่นใจ: {entity['confidence']:.4f}]")
        
        print(f"\nข้อความที่ทำเครื่องหมาย:")
        print(handler.format_entities_for_display(text, entities))

def analyze_entity_patterns():
    handler = NERHandler()
    
    print("\nการวิเคราะห์รูปแบบชื่อเฉพาะต่างๆ:")
    
    example_texts = {
        "ข่าว": "นายกรัฐมนตรีเดินทางเยือนประเทศญี่ปุ่นเมื่อวันที่ 15 มกราคม 2567",
        "ธุรกิจ": "บริษัท ABC จำกัด รายงานกำไรสุทธิ 500 ล้านบาทในไตรมาส 1",
        "กีฬา": "ทีมเชลซีเอาชนะแมนเชสเตอร์ซิตี้ 2-1 ที่สนามสแตมฟอร์ดบริดจ์",
        "บันเทิง": "ลิซ่า BLACKPINK จะจัดคอนเสิร์ตที่ราชมังคลากีฬาสถาน"
    }
    
    for category, text in example_texts.items():
        print(f"\nหมวด: {category}")
        print(f"ข้อความ: {text}")
        
        entities = handler.identify_entities(text)
        print("\nชื่อเฉพาะที่พบ:")
        for entity in entities:
            print(f"- {entity['text']} ({entity['type']}) [ความมั่นใจ: {entity['confidence']:.4f}]")
        
        print(f"\nข้อความที่ทำเครื่องหมาย:")
        print(handler.format_entities_for_display(text, entities))
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลระบุชื่อเฉพาะ...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze different entity patterns
    analyze_entity_patterns()