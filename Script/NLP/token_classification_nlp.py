import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from DatasetNLP.token_classification_nlp_dataset import token_classification_data

class TokenClassifier:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=7  # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
        )
        self.model.eval()
        
        # Define label mapping
        self.id2label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    def classify_tokens(self, text):
        # Tokenize text
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in tokens.items() if k != "offset_mapping"})
        
        # Get predicted labels
        predictions = torch.argmax(outputs.logits, dim=2)
        
        # Get token-level probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=2)
        confidence_scores = torch.max(probabilities, dim=2).values
        
        # Convert predictions to labels
        predicted_labels = [self.id2label[pred.item()] for pred in predictions[0]]
        
        # Get original tokens
        original_tokens = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        
        # Combine results
        results = []
        for token, label, conf in zip(original_tokens, predicted_labels, confidence_scores[0]):
            if token in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                continue
            results.append({
                "token": token,
                "label": label,
                "confidence": conf.item()
            })
        
        return results

def evaluate_model(classifier, test_data):
    all_results = []
    correct_predictions = 0
    total_predictions = 0
    
    for example in test_data:
        # Join tokens into text
        text = " ".join(example["tokens"])
        
        # Get predictions
        predictions = classifier.classify_tokens(text)
        
        # Compare with ground truth
        for i, (pred, true_token, true_label) in enumerate(zip(predictions, example["tokens"], example["ner_tags"])):
            result = {
                "token": true_token,
                "predicted_label": pred["label"],
                "true_label": true_label,
                "confidence": pred["confidence"],
                "is_correct": pred["label"] == true_label
            }
            all_results.append(result)
            
            if result["is_correct"]:
                correct_predictions += 1
            total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return all_results, accuracy

def demonstrate_usage():
    # Initialize classifier
    classifier = TokenClassifier()
    
    # Evaluate on test data
    print("ทดสอบโมเดลกับชุดข้อมูลทดสอบ:")
    results, accuracy = evaluate_model(classifier, token_classification_data)
    
    # Print results
    print("\nผลการทดสอบ:")
    current_sentence = []
    for result in results:
        current_sentence.append(result)
        
        # Print when we reach end of sentence or last result
        if result["token"] in [".", "?", "!"] or result == results[-1]:
            # Print sentence
            text = " ".join([r["token"] for r in current_sentence])
            print(f"\nประโยค: {text}")
            print("การวิเคราะห์:")
            for r in current_sentence:
                print(f"คำ: {r['token']}")
                print(f"  ป้ายที่ทำนาย: {r['predicted_label']}")
                print(f"  ป้ายที่ถูกต้อง: {r['true_label']}")
                print(f"  ความมั่นใจ: {r['confidence']:.4f}")
                print(f"  ถูกต้อง: {'ใช่' if r['is_correct'] else 'ไม่ใช่'}")
            current_sentence = []
    
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    classifier = TokenClassifier()
    
    print("\nทดสอบการระบุชนิดคำแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ประโยค (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        results = classifier.classify_tokens(text)
        
        print("\nผลการวิเคราะห์:")
        for result in results:
            print(f"คำ: {result['token']}")
            print(f"  ประเภท: {result['label']}")
            print(f"  ความมั่นใจ: {result['confidence']:.4f}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลระบุชนิดคำ...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()