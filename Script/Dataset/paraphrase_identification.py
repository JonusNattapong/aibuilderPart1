import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import pandas as pd
import os
from Dataset.paraphrase_identification_dataset import paraphrase_data

class ParaphraseHandler:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model for paraphrase detection
        self.detector_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        
        # Model for paraphrase generation
        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.detector_model.eval()
        self.generator_model.eval()
    
    def check_paraphrase(self, sentence1, sentence2):
        # Tokenize input pair
        inputs = self.tokenizer(
            sentence1,
            sentence2,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        is_paraphrase = torch.argmax(probabilities, dim=1).item() == 1
        confidence = probabilities[0][1].item() if is_paraphrase else probabilities[0][0].item()
        
        return {
            'is_paraphrase': is_paraphrase,
            'confidence': confidence,
            'similarity_score': self.calculate_similarity(sentence1, sentence2)
        }
    
    def generate_paraphrase(self, text, num_variations=3, max_length=128):
        # Prepare input with instruction
        input_text = f"แปลงประโยคต่อไปนี้เป็นประโยคที่มีความหมายเหมือนกัน: {text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Generate paraphrases
        outputs = self.generator_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_variations,
            num_beams=num_variations * 2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode and clean up generated paraphrases
        paraphrases = []
        for output in outputs:
            paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
            paraphrases.append({
                'text': paraphrase,
                'similarity': self.calculate_similarity(text, paraphrase)
            })
        
        return sorted(paraphrases, key=lambda x: x['similarity'], reverse=True)
    
    def calculate_similarity(self, text1, text2):
        # Get embeddings
        with torch.no_grad():
            inputs1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
            
            embedding1 = self.detector_model(**inputs1).logits
            embedding2 = self.detector_model(**inputs2).logits
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()
    
    def analyze_differences(self, text1, text2):
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Find common and different words
        common = words1.intersection(words2)
        unique1 = words1 - words2
        unique2 = words2 - words1
        
        return {
            'common_words': list(common),
            'unique_to_first': list(unique1),
            'unique_to_second': list(unique2),
            'word_overlap_ratio': len(common) / max(len(words1), len(words2))
        }

def evaluate_model():
    handler = ParaphraseHandler()
    results = []
    
    for item in paraphrase_data:
        # Check if sentences are paraphrases
        check_result = handler.check_paraphrase(
            item['sentence1'],
            item['sentence2']
        )
        
        # Analyze differences
        diff_analysis = handler.analyze_differences(
            item['sentence1'],
            item['sentence2']
        )
        
        # Generate alternative paraphrase
        generated = handler.generate_paraphrase(item['sentence1'], num_variations=1)[0]
        
        results.append({
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'expected_label': item['label'],
            'predicted_label': 1 if check_result['is_paraphrase'] else 0,
            'confidence': check_result['confidence'],
            'similarity_score': check_result['similarity_score'],
            'difference_analysis': diff_analysis,
            'generated_paraphrase': generated
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการตรวจจับและสร้างประโยคที่มีความหมายเหมือนกัน:\n")
    
    results = evaluate_model()
    
    # Group results by correctness
    correct = [r for r in results if r['expected_label'] == r['predicted_label']]
    incorrect = [r for r in results if r['expected_label'] != r['predicted_label']]
    
    print("ผลการทำนายที่ถูกต้อง:")
    print("-" * 50)
    for result in correct[:5]:  # Show first 5 examples
        print(f"ประโยค 1: {result['sentence1']}")
        print(f"ประโยค 2: {result['sentence2']}")
        print(f"ผลการตรวจสอบ: {'เป็น' if result['predicted_label'] == 1 else 'ไม่เป็น'}ประโยคที่มีความหมายเหมือนกัน")
        print(f"ความมั่นใจ: {result['confidence']:.4f}")
        print(f"คะแนนความคล้ายคลึง: {result['similarity_score']:.4f}")
        print("\nการวิเคราะห์ความแตกต่าง:")
        print(f"อัตราส่วนคำที่ซ้ำกัน: {result['difference_analysis']['word_overlap_ratio']:.2%}")
        print(f"คำที่เหมือนกัน: {', '.join(result['difference_analysis']['common_words'])}")
        print("\nประโยคที่สร้างขึ้นใหม่:")
        print(f"ข้อความ: {result['generated_paraphrase']['text']}")
        print(f"ความคล้ายคลึง: {result['generated_paraphrase']['similarity']:.4f}")
        print("-" * 50)
    
    # Calculate accuracy
    accuracy = len(correct) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    handler = ParaphraseHandler()
    
    print("\nทดสอบการตรวจจับและสร้างประโยคแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nเลือกโหมด:")
        print("1. ตรวจสอบประโยคคู่")
        print("2. สร้างประโยคใหม่")
        print("0. ออกจากโปรแกรม")
        
        choice = input("เลือกตัวเลข (0-2): ")
        
        if choice == "0":
            break
        elif choice == "1":
            print("\nใส่ประโยคที่ 1:")
            sent1 = input()
            print("ใส่ประโยคที่ 2:")
            sent2 = input()
            
            result = handler.check_paraphrase(sent1, sent2)
            diff_analysis = handler.analyze_differences(sent1, sent2)
            
            print("\nผลการวิเคราะห์:")
            print(f"เป็นประโยคที่มีความหมายเหมือนกัน: {'ใช่' if result['is_paraphrase'] else 'ไม่ใช่'}")
            print(f"ความมั่นใจ: {result['confidence']:.4f}")
            print(f"คะแนนความคล้ายคลึง: {result['similarity_score']:.4f}")
            print(f"อัตราส่วนคำที่ซ้ำกัน: {diff_analysis['word_overlap_ratio']:.2%}")
            
        elif choice == "2":
            print("\nใส่ประโยคที่ต้องการแปลง:")
            text = input()
            
            variations = handler.generate_paraphrase(text, num_variations=3)
            
            print("\nประโยคที่สร้างขึ้น:")
            for i, var in enumerate(variations, 1):
                print(f"\n{i}. {var['text']}")
                print(f"ความคล้ายคลึง: {var['similarity']:.4f}")

def analyze_paraphrase_patterns():
    handler = ParaphraseHandler()
    
    print("\nการวิเคราะห์รูปแบบการแปลงประโยค:")
    
    example_pairs = {
        "การเปลี่ยนคำ": {
            "original": "ฉันชอบอ่านหนังสือมาก",
            "variations": ["ฉันรักการอ่านหนังสือ", "ฉันเป็นคนรักการอ่าน"]
        },
        "การเรียงลำดับคำ": {
            "original": "วันนี้อากาศดีมาก",
            "variations": ["อากาศวันนี้ดีมาก", "ช่างเป็นวันที่อากาศดี"]
        },
        "การเปลี่ยนโครงสร้าง": {
            "original": "เขาทำงานหนักเพื่อความสำเร็จ",
            "variations": ["ความสำเร็จมาจากการทำงานหนักของเขา", "เขาประสบความสำเร็จเพราะทำงานหนัก"]
        }
    }
    
    for pattern, data in example_pairs.items():
        print(f"\nรูปแบบ: {pattern}")
        print(f"ประโยคต้นฉบับ: {data['original']}")
        
        for i, variation in enumerate(data['variations'], 1):
            result = handler.check_paraphrase(data['original'], variation)
            diff_analysis = handler.analyze_differences(data['original'], variation)
            
            print(f"\nรูปแบบที่ {i}: {variation}")
            print(f"ความคล้ายคลึง: {result['similarity_score']:.4f}")
            print(f"อัตราส่วนคำที่ซ้ำกัน: {diff_analysis['word_overlap_ratio']:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลตรวจจับและสร้างประโยค...")
    
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_dir = 'DataOutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Analyze paraphrase patterns
    analyze_paraphrase_patterns()