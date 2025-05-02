# -*- coding: utf-8 -*-
"""
Sample dataset for the Retail domain covering various NLP tasks.
Can also be run to generate retail_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd # Added import

# --- 1. Summarization ---
retail_summarization_data = [
    {
        "document": "รายงานยอดขายประจำสัปดาห์ของสาขา XYZ พบว่าสินค้ากลุ่มเครื่องสำอางมียอดขายสูงสุด รองลงมาคือกลุ่มเสื้อผ้าแฟชั่นสตรี สินค้าที่ขายดีที่สุดคือลิปสติกสีแดงรุ่นใหม่ และกางเกงยีนส์ทรงสกินนี่ โปรโมชั่นซื้อ 1 แถม 1 สำหรับสินค้ากลุ่มดูแลผิวได้รับความสนใจอย่างมาก แต่สินค้ากลุ่มเครื่องใช้ไฟฟ้ามียอดขายต่ำกว่าเป้าหมาย",
        "summary": "ยอดขายสาขา XYZ สัปดาห์นี้ เครื่องสำอางสูงสุด ตามด้วยเสื้อผ้าสตรี ลิปสติกแดงและยีนส์สกินนี่ขายดีสุด โปรฯ สกินแคร์ 1 แถม 1 ได้รับความสนใจ แต่เครื่องใช้ไฟฟ้าต่ำกว่าเป้า"
    },
    # ... more summarization examples
]

# --- 2. Open QA ---
retail_open_qa_data = [
    {
        "context": "นโยบายการคืนสินค้าของร้าน ABC ลูกค้าสามารถคืนสินค้าได้ภายใน 14 วันนับจากวันที่ซื้อ โดยสินค้าต้องอยู่ในสภาพสมบูรณ์พร้อมป้ายราคาและใบเสร็จรับเงิน ยกเว้นสินค้าลดราคาและสินค้าประเภทชุดชั้นในไม่สามารถคืนได้",
        "question": "สินค้าประเภทใดบ้างที่ไม่สามารถคืนได้ตามนโยบายของร้าน ABC?",
        "answer": "สินค้าลดราคาและสินค้าประเภทชุดชั้นในไม่สามารถคืนได้"
    },
    # ... more open QA examples
]

# --- 3. Close QA (Extractive) ---
retail_close_qa_data = [
    {
        "context": "โปรโมชั่นพิเศษสำหรับสมาชิก! ซื้อสินค้าครบ 1,000 บาท รับส่วนลดทันที 100 บาท และสะสมแต้มคูณสอง เฉพาะการซื้อสินค้าที่หน้าร้านเท่านั้น ตั้งแต่วันนี้ถึง 30 มิถุนายน",
        "question": "ต้องซื้อสินค้าครบเท่าไหร่จึงจะได้รับส่วนลด 100 บาท?",
        "answer_text": "1,000 บาท"
    },
    # ... more close QA examples
]

# --- 4. Classification ---
# Example labels: product_inquiry, order_status, complaint, promotion_query
retail_classification_data = [
    {"text": "รองเท้ารุ่นนี้มีสีดำไซส์ 40 ไหมคะ", "label": "product_inquiry"},
    {"text": "สั่งของไปเมื่อวาน อยากทราบสถานะการจัดส่ง", "label": "order_status"},
    {"text": "ได้รับสินค้าผิดขนาด ต้องการเปลี่ยนคืน", "label": "complaint"},
    {"text": "โปรโมชั่นลดราคามีถึงวันไหน", "label": "promotion_query"},
    # ... more classification examples
]

# --- 5. Creative Writing ---
retail_creative_writing_data = [
    {
        "prompt": "เขียนคำบรรยายสินค้าสำหรับเสื้อยืดคอลเลคชั่นใหม่ สไตล์มินิมอล",
        "generated_text": "เรียบง่ายแต่มีสไตล์ กับเสื้อยืดคอลเลคชั่นใหม่ล่าสุด เนื้อผ้าคอตตอน 100% นุ่มสบาย ระบายอากาศดีเยี่ยม ดีไซน์มินิมอล แมทช์ง่ายกับทุกลุค ใส่ได้ทุกวันไม่มีเบื่อ"
    },
    {
        "prompt": "เขียนโพสต์โปรโมทแคมเปญลดราคากลางปีบนโซเชียลมีเดีย",
        "generated_text": "🔥 Mid Year Sale มาแล้ว! ลดกระหน่ำสูงสุด 70% 🔥 ขนทัพสินค้าแบรนด์ดังมาลดราคาพิเศษ ช้อปจุใจได้แล้ววันนี้ - 31 ก.ค. นี้เท่านั้น! รีบเลยก่อนของหมด! #MidYearSale #ลดราคา #โปรโมชั่น"
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming ---
retail_brainstorming_data = [
    {
        "topic": "ไอเดียจัดกิจกรรมส่งเสริมการขายช่วงเทศกาลสงกรานต์",
        "ideas": [
            "จัดโปรโมชั่นซื้อสินค้าธีมสงกรานต์รับส่วนลดพิเศษ",
            "กิจกรรมสรงน้ำพระในร้าน",
            "แจกซองกันน้ำลายพิเศษเมื่อซื้อครบตามกำหนด",
            "จัดมุมถ่ายรูปธีมสงกรานต์",
            "ประกวดแต่งกายชุดไทยมารับส่วนลด"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA ---
retail_mcq_data = [
    {
        "context": "ร้านสะดวกซื้อเปิดให้บริการตลอด 24 ชั่วโมงทุกวัน ไม่มีวันหยุด เพื่ออำนวยความสะดวกให้แก่ลูกค้าในการซื้อสินค้าอุปโภคบริโภคที่จำเป็นได้ตลอดเวลา",
        "question": "ร้านสะดวกซื้อปิดให้บริการวันใดบ้าง?",
        "choices": ["วันอาทิตย์", "วันหยุดนักขัตฤกษ์", "วันปีใหม่", "ไม่มีวันหยุด"],
        "answer_index": 3 # Index of "ไม่มีวันหยุด"
    },
    # ... more multiple choice QA examples
]

# Combine all data
retail_domain_data = {
    "summarization": retail_summarization_data,
    "open_qa": retail_open_qa_data,
    "close_qa": retail_close_qa_data,
    "classification": retail_classification_data,
    "creative_writing": retail_creative_writing_data,
    "brainstorming": retail_brainstorming_data,
    "multiple_choice_qa": retail_mcq_data,
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
    print("--- Retail Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{retail_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{retail_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{retail_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{retail_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{retail_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{retail_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{retail_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "retail"
    for task, data in retail_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "retail_data.parquet"
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
