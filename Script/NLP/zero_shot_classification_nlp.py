import torch
from transformers import AutoTokenizer, AutoModel
from DatasetNLP.zero_shot_classification_nlp_dataset import zero_shot_data
import pandas as pd

class ZeroShotClassifier:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text):
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        return embeddings
    
    def create_label_embedding(self, label):
        # Create a template sentence for the label
        template = f"นี่คือเรื่องเกี่ยวกับ{label}"
        return self.get_embedding(template)
    
    def classify(self, text, candidate_labels, return_all_scores=False):
        # Get text embedding
        text_embedding = self.get_embedding(text)
        
        # Get label embeddings and compute similarities
        scores = []
        for label in candidate_labels:
            label_embedding = self.create_label_embedding(label)
            similarity = torch.nn.functional.cosine_similarity(text_embedding, label_embedding)
            scores.append({
                'label': label,
                'score': similarity.item()
            })
        
        # Sort by score
        scores = sorted(scores, key=lambda x: x['score'], reverse=True)
        
        if return_all_scores:
            return scores
        else:
            return scores[0]['label']

def evaluate_model():
    classifier = ZeroShotClassifier()
    results = []
    
    for example in zero_shot_data:
        # Get predictions with scores
        predictions = classifier.classify(
            example['sequence'],
            example['candidate_labels'],
            return_all_scores=True
        )
        
        # Check if top prediction matches expected label
        is_correct = predictions[0]['label'] == example['expected_label']
        
        results.append({
            'text': example['sequence'],
            'predictions': predictions,
            'expected_label': example['expected_label'],
            'is_correct': is_correct
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการจำแนกประเภทแบบ Zero-shot:\n")
    
    results = evaluate_model()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print(f"ข้อความ: {result['text']}")
        print(f"ประเภทที่ถูกต้อง: {result['expected_label']}")
        
        print("\nผลการทำนาย:")
        for pred in result['predictions']:
            print(f"- {pred['label']}: {pred['score']:.4f}")
        
        print(f"ทำนายถูกต้อง: {'ใช่' if result['is_correct'] else 'ไม่ใช่'}")
        print("-" * 50)
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r['is_correct']) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    classifier = ZeroShotClassifier()
    
    print("\nทดสอบการจำแนกประเภทแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ข้อความที่ต้องการจำแนก (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        print("\nใส่ประเภทที่เป็นไปได้ (คั่นด้วยเครื่องหมาย ,):")
        labels_input = input()
        candidate_labels = [label.strip() for label in labels_input.split(',')]
        
        if len(candidate_labels) < 2:
            print("กรุณาใส่อย่างน้อย 2 ประเภท")
            continue
        
        # Get predictions with scores
        predictions = classifier.classify(text, candidate_labels, return_all_scores=True)
        
        print("\nผลการจำแนกประเภท:")
        for pred in predictions:
            print(f"- {pred['label']}: {pred['score']:.4f}")

def custom_classification_example():
    classifier = ZeroShotClassifier()
    
    print("\nตัวอย่างการจำแนกประเภทแบบกำหนดเอง:")
    
    # Example 1: Sentiment Analysis
    text = "สินค้าคุณภาพดีมาก ราคาไม่แพง แนะนำให้ซื้อ"
    sentiment_labels = ["รีวิวเชิงบวก", "รีวิวเชิงลบ", "รีวิวเป็นกลาง"]
    
    print("\nการวิเคราะห์ความรู้สึก:")
    print(f"ข้อความ: {text}")
    predictions = classifier.classify(text, sentiment_labels, return_all_scores=True)
    for pred in predictions:
        print(f"- {pred['label']}: {pred['score']:.4f}")
    
    # Example 2: Topic Classification
    text = "วิธีปลูกต้นไม้ในพื้นที่จำกัด และการดูแลรักษา"
    topic_labels = ["การเกษตร", "การตกแต่งบ้าน", "สิ่งแวดล้อม", "งานอดิเรก"]
    
    print("\nการจำแนกหัวข้อ:")
    print(f"ข้อความ: {text}")
    predictions = classifier.classify(text, topic_labels, return_all_scores=True)
    for pred in predictions:
        print(f"- {pred['label']}: {pred['score']:.4f}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลจำแนกประเภทแบบ Zero-shot...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Show custom classification examples
    custom_classification_example()