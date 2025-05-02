# -*- coding: utf-8 -*-
"""
Sample dataset for the Finance domain covering various NLP tasks.
Can also be run to generate finance_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd # Added import

# --- 1. Summarization ---
finance_summarization_data = [
    {
        "document": "ตลาดหุ้นไทยวันนี้ปิดที่ 1,650.50 จุด เพิ่มขึ้น 5.20 จุด (+0.32%) มูลค่าการซื้อขายรวม 65,000 ล้านบาท นักลงทุนต่างชาติซื้อสุทธิ 1,200 ล้านบาท ขณะที่นักลงทุนสถาบันขายสุทธิ 800 ล้านบาท หุ้นกลุ่มพลังงานและธนาคารปรับตัวขึ้นนำตลาด จากราคาน้ำมันดิบที่สูงขึ้นและคาดการณ์ผลประกอบการที่ดี",
        "summary": "หุ้นไทยปิดบวก 5.20 จุด ที่ 1,650.50 มูลค่าซื้อขาย 6.5 หมื่นล้าน ต่างชาติซื้อสุทธิ 1.2 พันล้าน สถาบันขายสุทธิ 800 ล้าน กลุ่มพลังงานและแบงก์นำตลาด"
    },
    # ... more summarization examples
]

# --- 2. Open QA ---
finance_open_qa_data = [
    {
        "context": "กองทุนรวม RMF (Retirement Mutual Fund) เป็นกองทุนรวมประเภทหนึ่งที่มีวัตถุประสงค์เพื่อส่งเสริมการออมระยะยาวไว้ใช้จ่ายยามเกษียณอายุ ผู้ลงทุนสามารถนำเงินลงทุนไปลดหย่อนภาษีได้ตามเงื่อนไขที่กรมสรรพากรกำหนด โดยต้องถือหน่วยลงทุนไว้อย่างน้อย 5 ปี และขายได้เมื่ออายุครบ 55 ปีบริบูรณ์",
        "question": "เงื่อนไขหลักในการขายกองทุน RMF คืออะไร?",
        "answer": "ต้องถือหน่วยลงทุนไว้อย่างน้อย 5 ปี และผู้ลงทุนต้องมีอายุครบ 55 ปีบริบูรณ์จึงจะสามารถขายคืนได้โดยไม่ผิดเงื่อนไขทางภาษี"
    },
    # ... more open QA examples
]

# --- 3. Close QA (Extractive) ---
finance_close_qa_data = [
    {
        "context": "อัตราดอกเบี้ยนโยบายถูกกำหนดโดยคณะกรรมการนโยบายการเงิน (กนง.) ซึ่งเป็นเครื่องมือสำคัญในการดูแลเสถียรภาพทางเศรษฐกิจและการเงินของประเทศ การปรับขึ้นหรือลงอัตราดอกเบี้ยมีผลต่อต้นทุนทางการเงินของภาคธุรกิจและประชาชน",
        "question": "ใครเป็นผู้กำหนดอัตราดอกเบี้ยนโยบาย?",
        "answer_text": "คณะกรรมการนโยบายการเงิน (กนง.)"
    },
    # ... more close QA examples
]

# --- 4. Classification ---
# Example labels: investment_advice, loan_application, insurance_query, market_news
finance_classification_data = [
    {"text": "อยากลงทุนในหุ้นปันผลสูง ควรเลือกตัวไหนดี", "label": "investment_advice"},
    {"text": "ต้องการยื่นกู้สินเชื่อบ้าน ต้องใช้เอกสารอะไรบ้าง", "label": "loan_application"},
    {"text": "ประกันสุขภาพแบบไหนคุ้มครองค่าห้องสูงสุด", "label": "insurance_query"},
    {"text": "เฟดประกาศคงอัตราดอกเบี้ยในการประชุมล่าสุด", "label": "market_news"},
    # ... more classification examples
]

# --- 5. Creative Writing ---
finance_creative_writing_data = [
    {
        "prompt": "เขียนบทความสั้นๆ เชิญชวนให้คนรุ่นใหม่เริ่มวางแผนการเงินเพื่ออนาคต",
        "generated_text": "อนาคตไม่แน่นอน แต่การเงินที่มั่นคงสร้างได้! เริ่มต้นวางแผนการเงินตั้งแต่วันนี้ ออมก่อนใช้ ลงทุนอย่างชาญฉลาด เพื่ออิสรภาพทางการเงินในวันหน้า อย่ารอช้า เริ่มต้นสร้างความมั่งคั่งให้ตัวเองตั้งแต่วันนี้!"
    },
    {
        "prompt": "เขียนสโลแกนสำหรับแอปพลิเคชันลงทุนในกองทุนรวม",
        "generated_text": "ลงทุนง่าย แค่ปลายนิ้ว กองทุนรวมเพื่ออนาคตในมือคุณ"
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming ---
finance_brainstorming_data = [
    {
        "topic": "ไอเดียผลิตภัณฑ์ทางการเงินสำหรับกลุ่ม Freelance",
        "ideas": [
            "สินเชื่อส่วนบุคคลดอกเบี้ยพิเศษสำหรับฟรีแลนซ์",
            "ประกันสุขภาพที่ครอบคลุมช่วงว่างงาน",
            "กองทุนสำรองเลี้ยงชีพภาคสมัครใจ",
            "โปรแกรมช่วยวางแผนภาษีสำหรับผู้มีรายได้ไม่แน่นอน",
            "บัตรเครดิตพร้อมสิทธิประโยชน์สำหรับฟรีแลนซ์"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA ---
finance_mcq_data = [
    {
        "context": "เงินเฟ้อคือภาวะที่ระดับราคาสินค้าและบริการโดยทั่วไปเพิ่มสูงขึ้นอย่างต่อเนื่อง ทำให้มูลค่าของเงินลดลง หรืออำนาจซื้อของผู้บริโภคลดลงเมื่อเทียบกับช่วงเวลาก่อนหน้า",
        "question": "ผลกระทบหลักของเงินเฟ้อคืออะไร?",
        "choices": ["มูลค่าเงินเพิ่มขึ้น", "อำนาจซื้อของผู้บริโภคลดลง", "ราคาสินค้าลดลง", "เศรษฐกิจเติบโตเร็วขึ้น"],
        "answer_index": 1 # Index of "อำนาจซื้อของผู้บริโภคลดลง"
    },
    # ... more multiple choice QA examples
]

# Combine all data
finance_domain_data = {
    "summarization": finance_summarization_data,
    "open_qa": finance_open_qa_data,
    "close_qa": finance_close_qa_data,
    "classification": finance_classification_data,
    "creative_writing": finance_creative_writing_data,
    "brainstorming": finance_brainstorming_data,
    "multiple_choice_qa": finance_mcq_data,
}

# Function to preprocess data into a standard format for Parquet (copied for simplicity)
def preprocess_data_for_parquet(domain_name, task_name, data_list):
    processed = []
    for item in data_list:
        input_text = ""
        target_text = ""
        prefix = f"{domain_name} {task_name}: "

        if task_name == "summarization":
            input_text = prefix + item.get("document", "")
            target_text = item.get("summary", "")
        elif task_name == "open_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer", "")
        elif task_name == "close_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer_text", "")
        elif task_name == "classification":
            input_text = prefix + item.get("text", "")
            target_text = item.get("label", "")
        elif task_name == "creative_writing":
            input_text = prefix + item.get("prompt", "")
            target_text = item.get("generated_text", "")
        elif task_name == "brainstorming":
            input_text = prefix + item.get("topic", "")
            target_text = "\n".join(item.get("ideas", []))
        elif task_name == "multiple_choice_qa":
            choices_str = " | ".join(item.get("choices", []))
            input_text = prefix + f"question: {item.get('question', '')} choices: {choices_str} context: {item.get('context', '')}"
            answer_idx = item.get("answer_index")
            if answer_idx is not None and item.get("choices"):
                target_text = item["choices"][answer_idx]

        if input_text and target_text:
            processed.append({
                "domain": domain_name,
                "task": task_name,
                "input_text": input_text,
                "target_text": target_text
            })
    return processed


if __name__ == '__main__':
    print("--- Finance Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{finance_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{finance_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{finance_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{finance_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{finance_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{finance_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{finance_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "finance"
    for task, data in finance_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "finance_data.parquet"
        # Define output directory relative to the script's location
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.abspath(os.path.join(script_dir, '..', 'DataOutput')) # Go up one level, then into DataOutput
        # Create DataOutput directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        try:
            df.to_parquet(output_path, index=False)
            print(f"Successfully generated {output_filename} at {output_path}")
            print(f"DataFrame Info:\n{df.info()}")
            print(f"\nFirst 5 rows:\n{df.head().to_string()}")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
            print("Please ensure 'pandas' and 'pyarrow' are installed (`pip install pandas pyarrow`)")
    else:
        print("No records processed, Parquet file not generated.")
