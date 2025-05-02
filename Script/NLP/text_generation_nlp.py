import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from DatasetNLP.text_generation_nlp_dataset import text_generation_prompts

class TextGenerator:
    def __init__(self, model_name="airesearch/wangchanberta-base-att-spm-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
    
    def generate_text(self, prompt, max_length=100, num_return_sequences=1, 
                     temperature=0.7, top_k=50, top_p=0.9):
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate text
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            no_repeat_ngram_size=2
        )
        
        # Decode and return generated sequences
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the generated text if it appears at the start
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            generated_texts.append(generated_text)
        
        return generated_texts

def demonstrate_usage():
    # Initialize generator
    generator = TextGenerator()
    
    print("ทดสอบการสร้างข้อความจากประโยคเริ่มต้น:\n")
    
    for i, prompt in enumerate(text_generation_prompts, 1):
        print(f"\nประโยคเริ่มต้น {i}: {prompt}")
        print("การสร้างข้อความ:")
        
        # Generate multiple variations
        generated_texts = generator.generate_text(
            prompt,
            max_length=150,
            num_return_sequences=3,
            temperature=0.7
        )
        
        for j, text in enumerate(generated_texts, 1):
            print(f"\nรูปแบบที่ {j}:")
            print(f"{prompt}{text}")
        print("-" * 50)

def run_interactive_demo():
    generator = TextGenerator()
    
    print("\nทดสอบการสร้างข้อความแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nใส่ประโยคเริ่มต้น (หรือพิมพ์ 'exit' เพื่อออก):")
        prompt = input()
        if prompt.lower() == 'exit':
            break
        
        print("\nกำลังสร้างข้อความ...")
        generated_texts = generator.generate_text(
            prompt,
            max_length=150,
            num_return_sequences=3,
            temperature=0.7
        )
        
        print("\nผลลัพธ์:")
        for i, text in enumerate(generated_texts, 1):
            print(f"\nรูปแบบที่ {i}:")
            print(f"{prompt}{text}")

def experiment_with_parameters():
    generator = TextGenerator()
    test_prompt = "กาลครั้งหนึ่งนานมาแล้ว ในป่าลึกแห่งหนึ่ง มี"
    
    print("ทดลองปรับพารามิเตอร์การสร้างข้อความ:\n")
    
    # Test different temperatures
    temperatures = [0.3, 0.7, 1.0]
    for temp in temperatures:
        print(f"\nTemperature = {temp} (ความหลากหลายของการสร้างข้อความ):")
        texts = generator.generate_text(test_prompt, temperature=temp)
        print(f"{test_prompt}{texts[0]}")
    
    # Test different top_k values
    top_k_values = [10, 50, 100]
    for k in top_k_values:
        print(f"\nTop-k = {k} (จำนวนคำที่พิจารณาในแต่ละขั้นตอน):")
        texts = generator.generate_text(test_prompt, top_k=k)
        print(f"{test_prompt}{texts[0]}")
    
    # Test different top_p values
    top_p_values = [0.5, 0.9, 0.95]
    for p in top_p_values:
        print(f"\nTop-p = {p} (ความหลากหลายของคำศัพท์):")
        texts = generator.generate_text(test_prompt, top_p=p)
        print(f"{test_prompt}{texts[0]}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลสร้างข้อความ...")
    
    # Basic demonstration
    demonstrate_usage()
    
    # Parameter experimentation
    experiment_with_parameters()
    
    # Interactive demo
    run_interactive_demo()