# -*- coding: utf-8 -*-
"""
Sample dataset for the Legal domain covering various NLP tasks.
Can also be run to generate legal_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd # Added import

# --- 1. Summarization ---
legal_summarization_data = [
    {
        "document": "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 1448 บัญญัติว่า การสมรสจะทำได้ต่อเมื่อชายและหญิงมีอายุสิบเจ็ดปีบริบูรณ์แล้ว แต่ในกรณีมีเหตุอันสมควร ศาลอาจอนุญาตให้ทำการสมรสก่อนนั้นได้ การสมรสที่ฝ่าฝืนเงื่อนไขเรื่องอายุถือเป็นโมฆียะ ซึ่งหมายความว่าอาจถูกบอกล้างโดยผู้มีส่วนได้เสียหรืออัยการได้",
        "summary": "กฎหมายกำหนดให้ชายหญิงต้องอายุ 17 ปีบริบูรณ์จึงจะสมรสได้ หากต่ำกว่านั้นต้องได้รับอนุญาตจากศาล การสมรสที่ฝ่าฝืนเงื่อนไขอายุเป็นโมฆียะ สามารถถูกบอกล้างได้"
    },
    # ... more summarization examples
]

# --- 2. Open QA ---
legal_open_qa_data = [
    {
        "context": "สัญญาเช่าซื้อเป็นสัญญาที่เจ้าของทรัพย์สินเอาทรัพย์สินออกให้เช่า และให้คำมั่นว่าจะขายทรัพย์สินนั้น หรือว่าจะให้ทรัพย์สินนั้นตกเป็นสิทธิแก่ผู้เช่า โดยเงื่อนไขที่ผู้เช่าได้ใช้เงินเป็นจำนวนเท่านั้นเท่านี้คราว ผู้เช่าซื้อมีหน้าที่ชำระค่าเช่าซื้อให้ครบถ้วนตามสัญญา และดูแลรักษาทรัพย์สินที่เช่าซื้อเสมือนวิญญูชนจะพึงสงวนทรัพย์สินของตนเอง",
        "question": "หน้าที่หลักของผู้เช่าซื้อตามสัญญาเช่าซื้อคืออะไร?",
        "answer": "ผู้เช่าซื้อมีหน้าที่ชำระค่าเช่าซื้อให้ครบตามสัญญา และต้องดูแลรักษาทรัพย์สินที่เช่าซื้ออย่างดีเหมือนเป็นของตนเอง"
    },
    # ... more open QA examples
]

# --- 3. Close QA (Extractive) ---
legal_close_qa_data = [
    {
        "context": "ตามพระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 ผู้ควบคุมข้อมูลส่วนบุคคลต้องแจ้งวัตถุประสงค์ในการเก็บรวบรวม ใช้ หรือเปิดเผยข้อมูลส่วนบุคคลให้เจ้าของข้อมูลทราบก่อนหรือในขณะเก็บรวบรวมข้อมูล เว้นแต่จะมีข้อยกเว้นตามกฎหมาย",
        "question": "ผู้ควบคุมข้อมูลต้องแจ้งวัตถุประสงค์ในการเก็บข้อมูลเมื่อใด?",
        "answer_text": "ก่อนหรือในขณะเก็บรวบรวมข้อมูล"
    },
    # ... more close QA examples
]

# --- 4. Classification ---
# Example labels: contract_law, criminal_law, family_law, intellectual_property
legal_classification_data = [
    {"text": "การผิดสัญญาซื้อขายที่ดินต้องทำอย่างไร", "label": "contract_law"},
    {"text": "ข้อหาลักทรัพย์ในเวลากลางคืนมีโทษเท่าไหร่", "label": "criminal_law"},
    {"text": "ต้องการฟ้องหย่าและเรียกค่าเลี้ยงดูบุตร", "label": "family_law"},
    {"text": "มีคนลอกเลียนแบบสินค้าและเครื่องหมายการค้า", "label": "intellectual_property"},
    # ... more classification examples
]

# --- 5. Creative Writing ---
legal_creative_writing_data = [
    {
        "prompt": "เขียนบทสรุปสำหรับผู้บริหารเกี่ยวกับประเด็นทางกฎหมายที่สำคัญในการควบรวมกิจการ",
        "generated_text": "บทสรุปสำหรับผู้บริหาร: การควบรวมกิจการครั้งนี้มีประเด็นทางกฎหมายที่ต้องพิจารณาหลักๆ ได้แก่ การตรวจสอบสถานะกิจการ (Due Diligence) อย่างละเอียด การปฏิบัติตามกฎหมายแข่งขันทางการค้า การจัดการสัญญาและภาระผูกพันเดิมของทั้งสองบริษัท รวมถึงการดำเนินการด้านแรงงานและพนักงานตามกฎหมายคุ้มครองแรงงาน..."
    },
    {
        "prompt": "เขียนคำโต้แย้งสั้นๆ ในนามจำเลยคดีหมิ่นประมาท ว่าข้อความที่กล่าวไปเป็นการแสดงความคิดเห็นโดยสุจริต",
        "generated_text": "ข้อความที่จำเลยกล่าวไปนั้น เป็นเพียงการแสดงความคิดเห็นโดยสุจริต ติชมด้วยความเป็นธรรม อันเป็นวิสัยของประชาชนย่อมกระทำได้ มิได้มีเจตนาใส่ความโจทก์ให้ได้รับความเสียหาย จึงไม่เป็นความผิดฐานหมิ่นประมาท"
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming ---
legal_brainstorming_data = [
    {
        "topic": "แนวทางการให้ความช่วยเหลือทางกฎหมายแก่ผู้มีรายได้น้อยในชุมชน",
        "ideas": [
            "จัดตั้งคลินิกกฎหมายเคลื่อนที่",
            "จัดอบรมให้ความรู้กฎหมายเบื้องต้นที่ประชาชนควรรู้",
            "ร่วมมือกับทนายอาสาและนักศึกษากฎหมาย",
            "สร้างแพลตฟอร์มออนไลน์ให้คำปรึกษาเบื้องต้นฟรี",
            "จัดทำเอกสารเผยแพร่ความรู้ทางกฎหมายที่เข้าใจง่าย"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA ---
legal_mcq_data = [
    {
        "context": "อายุความในคดีอาญาคือระยะเวลาที่กฎหมายกำหนดให้พนักงานอัยการต้องฟ้องผู้กระทำความผิดต่อศาล หากฟ้องคดีเมื่อพ้นกำหนดอายุความแล้ว ศาลจะต้องยกฟ้อง สำหรับความผิดลหุโทษ มีอายุความ 1 ปีนับแต่วันกระทำความผิด",
        "question": "ความผิดลหุโทษมีอายุความเท่าใด?",
        "choices": ["6 เดือน", "1 ปี", "3 ปี", "5 ปี"],
        "answer_index": 1 # Index of "1 ปี"
    },
    # ... more multiple choice QA examples
]

# Combine all data
legal_domain_data = {
    "summarization": legal_summarization_data,
    "open_qa": legal_open_qa_data,
    "close_qa": legal_close_qa_data,
    "classification": legal_classification_data,
    "creative_writing": legal_creative_writing_data,
    "brainstorming": legal_brainstorming_data,
    "multiple_choice_qa": legal_mcq_data,
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
    print("--- Legal Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{legal_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{legal_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{legal_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{legal_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{legal_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{legal_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{legal_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "legal"
    for task, data in legal_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "legal_data.parquet"
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
