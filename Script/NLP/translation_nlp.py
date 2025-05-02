import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from DatasetNLP.translation_nlp_dataset import translation_data
import pandas as pd
from sklearn.model_selection import train_test_split

class Translator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-th-en"):
        self.th_en_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.th_en_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # For English to Thai, we'll use the reverse model
        self.en_th_model_name = "Helsinki-NLP/opus-mt-en-th"
        self.en_th_tokenizer = AutoTokenizer.from_pretrained(self.en_th_model_name)
        self.en_th_model = AutoModelForSeq2SeqLM.from_pretrained(self.en_th_model_name)
        
        # Set models to evaluation mode
        self.th_en_model.eval()
        self.en_th_model.eval()
        
        # Create translation pipelines
        self.th_en_pipeline = pipeline(
            "translation",
            model=self.th_en_model,
            tokenizer=self.th_en_tokenizer
        )
        
        self.en_th_pipeline = pipeline(
            "translation",
            model=self.en_th_model,
            tokenizer=self.en_th_tokenizer
        )
    
    def translate_th_to_en(self, text):
        # Translate Thai to English
        result = self.th_en_pipeline(text)
        return result[0]['translation_text']
    
    def translate_en_to_th(self, text):
        # Translate English to Thai
        result = self.en_th_pipeline(text)
        return result[0]['translation_text']
    
    def batch_translate(self, texts, source_lang):
        if source_lang == "th":
            return [self.translate_th_to_en(text) for text in texts]
        else:
            return [self.translate_en_to_th(text) for text in texts]

def evaluate_translations(translator, test_data):
    results = []
    
    for item in test_data:
        # Translate in both directions
        en_pred = translator.translate_th_to_en(item['th'])
        th_pred = translator.translate_en_to_th(item['en'])
        
        results.append({
            'th_original': item['th'],
            'en_original': item['en'],
            'th_to_en_pred': en_pred,
            'en_to_th_pred': th_pred,
            'th_to_en_match': en_pred.lower() == item['en'].lower(),
            'en_to_th_match': th_pred == item['th']
        })
    
    return pd.DataFrame(results)

def demonstrate_usage():
    # Initialize translator
    translator = Translator()
    
    # Split data into train/test
    test_data = translation_data  # Using all data for testing since we're using a pre-trained model
    
    # Evaluate translations
    print("ทดสอบการแปลภาษา:\n")
    results_df = evaluate_translations(translator, test_data)
    
    # Print results
    for _, row in results_df.iterrows():
        print("การแปลไทย -> อังกฤษ:")
        print(f"ต้นฉบับ (ไทย): {row['th_original']}")
        print(f"คำแปล (อังกฤษ): {row['th_to_en_pred']}")
        print(f"เป้าหมาย (อังกฤษ): {row['en_original']}")
        print(f"ตรงกัน: {'ใช่' if row['th_to_en_match'] else 'ไม่ใช่'}")
        
        print("\nการแปลอังกฤษ -> ไทย:")
        print(f"ต้นฉบับ (อังกฤษ): {row['en_original']}")
        print(f"คำแปล (ไทย): {row['en_to_th_pred']}")
        print(f"เป้าหมาย (ไทย): {row['th_original']}")
        print(f"ตรงกัน: {'ใช่' if row['en_to_th_match'] else 'ไม่ใช่'}")
        print("-" * 50)
    
    # Calculate and print accuracy
    th_en_accuracy = results_df['th_to_en_match'].mean()
    en_th_accuracy = results_df['en_to_th_match'].mean()
    
    print(f"\nความแม่นยำการแปลไทย -> อังกฤษ: {th_en_accuracy:.2%}")
    print(f"ความแม่นยำการแปลอังกฤษ -> ไทย: {en_th_accuracy:.2%}")

def run_interactive_demo():
    translator = Translator()
    
    print("\nทดสอบการแปลภาษาแบบโต้ตอบ")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        print("\nเลือกทิศทางการแปล:")
        print("1. ไทย -> อังกฤษ")
        print("2. อังกฤษ -> ไทย")
        print("0. ออกจากโปรแกรม")
        
        choice = input("เลือกตัวเลข (0-2): ")
        
        if choice == "0":
            break
        elif choice not in ["1", "2"]:
            print("กรุณาเลือก 0, 1 หรือ 2")
            continue
        
        print("\nใส่ข้อความที่ต้องการแปล (หรือพิมพ์ 'exit' เพื่อออก):")
        text = input()
        if text.lower() == 'exit':
            break
        
        if choice == "1":
            translated = translator.translate_th_to_en(text)
            print(f"\nคำแปล (อังกฤษ): {translated}")
        else:
            translated = translator.translate_en_to_th(text)
            print(f"\nคำแปล (ไทย): {translated}")

if __name__ == "__main__":
    print("เริ่มการทดสอบโมเดลแปลภาษา...")
    demonstrate_usage()
    
    # Run interactive demo
    run_interactive_demo()