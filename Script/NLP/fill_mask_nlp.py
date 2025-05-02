import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from DatasetNLP.fill_mask_nlp_dataset import fill_mask_sentences

class MaskFiller:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        
    def predict_masked_word(self, text, top_k=5):
        # Encode text
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Get mask token index
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits for masked position
        mask_token_logits = outputs.logits[0, mask_token_index, :]
        
        # Get top k token ids and probabilities
        top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1)
        top_k_token_ids = top_k_tokens.indices[0]
        top_k_probs = torch.softmax(top_k_tokens.values[0], dim=0)
        
        # Convert token ids to words
        predictions = []
        for token_id, prob in zip(top_k_token_ids, top_k_probs):
            token = self.tokenizer.decode([token_id])
            predictions.append({
                'token': token.strip(),
                'probability': prob.item()
            })
        
        return predictions

def demonstrate_usage():
    # Initialize mask filler
    mask_filler = MaskFiller()
    
    # Process each sentence
    print("ทดสอบการทำนายคำในช่องว่าง:\n")
    
    for sentence in fill_mask_sentences:
        print(f"ประโยค: {sentence}")
        predictions = mask_filler.predict_masked_word(sentence)
        
        print("คำที่น่าจะเป็นไปได้:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['token']} (ความน่าจะเป็น: {pred['probability']:.4f})")
        print()

def run_interactive_demo():
    mask_filler = MaskFiller()
    
    print("\nทดสอบการทำนายแบบโต้ตอบ")
    print("พิมพ์ประโยคโดยใส่ <mask> ในตำแหน่งที่ต้องการให้ทำนาย")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        text = input("\nใส่ประโยค: ")
        if text.lower() == 'exit':
            break
            
        if '<mask>' not in text:
            print("กรุณาใส่ <mask> ในประโยค")
            continue
            
        predictions = mask_filler.predict_masked_word(text)
        
        print("\nคำที่น่าจะเป็นไปได้:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['token']} (ความน่าจะเป็น: {pred['probability']:.4f})")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดล Fill-Mask...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()