import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from DatasetNLP.feature_extraction_nlp_dataset import feature_extraction_texts

class TextEmbedding:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode
        
    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling - take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def extract_features(self, texts, batch_size=32):
        all_embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize texts
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform mean pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Convert to numpy and store
            all_embeddings.append(sentence_embeddings.numpy())
        
        # Concatenate all embeddings
        return np.vstack(all_embeddings)

def demonstrate_usage():
    # Initialize text embedding
    embedder = TextEmbedding()
    
    # Extract features from example texts
    embeddings = embedder.extract_features(feature_extraction_texts)
    
    # Demonstrate embeddings shape and sample calculations
    print(f"Shape of embeddings: {embeddings.shape}")
    print("\nตัวอย่างการใช้งาน Embeddings:")
    
    # Calculate cosine similarity between first two texts
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    print(f"\nความคล้ายคลึงระหว่างข้อความ:")
    print(f"ข้อความ 1: {feature_extraction_texts[0]}")
    print(f"ข้อความ 2: {feature_extraction_texts[1]}")
    print(f"Cosine Similarity: {similarity:.4f}")
    
    # Find most similar pair
    n_texts = len(feature_extraction_texts)
    max_similarity = -1
    most_similar_pair = (0, 0)
    
    for i in range(n_texts):
        for j in range(i + 1, n_texts):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_similarity:
                max_similarity = sim
                most_similar_pair = (i, j)
    
    print(f"\nคู่ข้อความที่คล้ายกันมากที่สุด:")
    print(f"ข้อความ 1: {feature_extraction_texts[most_similar_pair[0]]}")
    print(f"ข้อความ 2: {feature_extraction_texts[most_similar_pair[1]]}")
    print(f"Cosine Similarity: {max_similarity:.4f}")
    
    # Save embeddings
    np.save('text_embeddings.npy', embeddings)
    print("\nบันทึก embeddings ไปยังไฟล์ 'text_embeddings.npy' แล้ว")

if __name__ == "__main__":
    print("เริ่มการสกัด features จากข้อความ...")
    demonstrate_usage()