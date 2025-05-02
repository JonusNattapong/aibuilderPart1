import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from DatasetNLP.sentence_similarity_nlp_dataset import sentence_similarity_data
import pandas as pd

class SentenceSimilarity:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform mean pooling
        sentence_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding
    
    def compute_similarity(self, sentence1, sentence2):
        # Get embeddings
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()

def evaluate_model():
    # Initialize similarity model
    similarity_model = SentenceSimilarity()
    
    results = []
    for item in sentence_similarity_data:
        # Compute similarity score
        predicted_score = similarity_model.compute_similarity(
            item['sentence1'],
            item['sentence2']
        )
        
        # Store results
        results.append({
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'true_score': item['similarity_score'],
            'predicted_score': predicted_score,
            'score_difference': abs(item['similarity_score'] - predicted_score)
        })
    
    return pd.DataFrame(results)

def demonstrate_usage():
    print("ทดสอบการวัดความคล้ายคลึงของประโยค:\n")
    
    results_df = evaluate_model()
    
    # Print results
    for _, row in results_df.iterrows():
        print(f"ประโยค 1: {row['sentence1']}")
        print(f"ประโยค 2: {row['sentence2']}")
        print(f"คะแนนที่กำหนด: {row['true_score']:.3f}")
        print(f"คะแนนที่ทำนาย: {row['predicted_score']:.3f}")
        print(f"ความต่าง: {row['score_difference']:.3f}")
        print("-" * 50)
    
    # Calculate overall metrics
    mean_difference = results_df['score_difference'].mean()
    print(f"\nค่าความต่างเฉลี่ย: {mean_difference:.3f}")

def run_interactive_demo():
    similarity_model = SentenceSimilarity()
    
    print("\nทดสอบการวัดความคล้ายคลึงแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ประโยคที่ 1 (หรือพิมพ์ 'exit' เพื่อออก):")
        sentence1 = input()
        if sentence1.lower() == 'exit':
            break
        
        print("ใส่ประโยคที่ 2:")
        sentence2 = input()
        
        similarity = similarity_model.compute_similarity(sentence1, sentence2)
        print(f"\nความคล้ายคลึง: {similarity:.3f}")
        
        # แสดงการตีความ
        if similarity > 0.8:
            print("การตีความ: ประโยคมีความหมายเหมือนกันมาก")
        elif similarity > 0.5:
            print("การตีความ: ประโยคมีความหมายคล้ายกันปานกลาง")
        else:
            print("การตีความ: ประโยคมีความหมายแตกต่างกัน")

def compare_multiple_sentences():
    similarity_model = SentenceSimilarity()
    
    print("\nทดสอบการเปรียบเทียบหลายประโยค")
    print("พิมพ์ 'done' เมื่อใส่ประโยคครบ หรือ 'exit' เพื่อออกจากโปรแกรม")
    
    sentences = []
    while True:
        print(f"\nใส่ประโยคที่ {len(sentences) + 1} (หรือพิมพ์ 'done' เมื่อเสร็จ, 'exit' เพื่อออก):")
        sentence = input()
        
        if sentence.lower() == 'exit':
            return
        elif sentence.lower() == 'done':
            if len(sentences) < 2:
                print("กรุณาใส่อย่างน้อย 2 ประโยค")
                continue
            break
        
        sentences.append(sentence)
    
    # Create similarity matrix
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            similarity = similarity_model.compute_similarity(sentences[i], sentences[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    # Print results
    print("\nตารางความคล้ายคลึง:")
    for i in range(n):
        print(f"\nประโยค {i+1}: {sentences[i]}")
    
    print("\nMatrix ความคล้ายคลึง:")
    print("     " + "     ".join(f"ประโยค {i+1:<3}" for i in range(n)))
    for i in range(n):
        print(f"ประโยค {i+1:<3}", end=" ")
        for j in range(n):
            print(f"{similarity_matrix[i][j]:.3f}    ", end="")
        print()

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลวัดความคล้ายคลึง...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Compare multiple sentences
    compare_multiple_sentences()