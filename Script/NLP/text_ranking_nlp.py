import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from DatasetNLP.text_ranking_nlp_dataset import text_ranking_data
from typing import List, Dict

class TextRanker:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text: str) -> torch.Tensor:
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Compute embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        return embeddings
    
    def compute_similarity(self, query_embedding: torch.Tensor, passage_embedding: torch.Tensor) -> float:
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(query_embedding, passage_embedding)
        return similarity.item()
    
    def rank_passages(self, query: str, passages: List[str]) -> List[Dict]:
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Get embeddings for all passages
        passage_similarities = []
        for passage in passages:
            passage_embedding = self.get_embedding(passage)
            similarity = self.compute_similarity(query_embedding, passage_embedding)
            passage_similarities.append({
                'passage': passage,
                'similarity': similarity
            })
        
        # Sort by similarity score
        ranked_passages = sorted(passage_similarities, key=lambda x: x['similarity'], reverse=True)
        return ranked_passages

def evaluate_model():
    ranker = TextRanker()
    results = []
    
    for example in text_ranking_data:
        # Combine positive and negative passages
        all_passages = [example['positive_passage']] + example['negative_passages']
        
        # Rank passages
        ranked_results = ranker.rank_passages(example['query'], all_passages)
        
        # Check if positive passage is ranked first
        is_correct = ranked_results[0]['passage'] == example['positive_passage']
        
        results.append({
            'query': example['query'],
            'ranked_passages': ranked_results,
            'is_correct': is_correct
        })
    
    return results

def demonstrate_usage():
    print("ทดสอบการจัดอันดับข้อความตามความเกี่ยวข้อง:\n")
    
    results = evaluate_model()
    
    for i, result in enumerate(results, 1):
        print(f"\nตัวอย่างที่ {i}:")
        print(f"คำค้น: {result['query']}")
        print("\nผลการจัดอันดับ:")
        for j, passage in enumerate(result['ranked_passages'], 1):
            print(f"\nอันดับที่ {j}:")
            print(f"ข้อความ: {passage['passage']}")
            print(f"คะแนนความเกี่ยวข้อง: {passage['similarity']:.4f}")
        
        print(f"\nจัดอันดับถูกต้อง: {'ใช่' if result['is_correct'] else 'ไม่ใช่'}")
        print("-" * 50)
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r['is_correct']) / len(results)
    print(f"\nความแม่นยำโดยรวม: {accuracy:.2%}")

def run_interactive_demo():
    ranker = TextRanker()
    
    print("\nทดสอบการจัดอันดับข้อความแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่คำค้น (หรือพิมพ์ 'exit' เพื่อออก):")
        query = input()
        if query.lower() == 'exit':
            break
        
        passages = []
        print("\nใส่ข้อความที่ต้องการจัดอันดับ (พิมพ์ 'done' เมื่อเสร็จ):")
        while True:
            print(f"\nข้อความที่ {len(passages) + 1}:")
            passage = input()
            if passage.lower() == 'done':
                if len(passages) < 2:
                    print("กรุณาใส่อย่างน้อย 2 ข้อความ")
                    continue
                break
            passages.append(passage)
        
        # Rank passages
        ranked_results = ranker.rank_passages(query, passages)
        
        print("\nผลการจัดอันดับ:")
        for i, result in enumerate(ranked_results, 1):
            print(f"\nอันดับที่ {i}:")
            print(f"ข้อความ: {result['passage']}")
            print(f"คะแนนความเกี่ยวข้อง: {result['similarity']:.4f}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลจัดอันดับข้อความ...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()