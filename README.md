# AI Builder Part 1: Thai NLP Dataset Generation and Utilities

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)

โปรเจกต์นี้รวบรวมเครื่องมือและสคริปต์สำหรับการสร้างชุดข้อมูล (Dataset) ภาษาไทยสำหรับงาน Natural Language Processing (NLP) ต่างๆ รวมถึงสคริปต์สาธิตการใช้งานโมเดลพื้นฐานสำหรับงานเหล่านั้น

## ✨ คุณสมบัติหลัก (Key Features)

*   **Dataset Generation:**
    *   สร้างชุดข้อมูลสำหรับงาน NLP หลากหลายประเภทโดยใช้ Large Language Models (LLM) เช่น DeepSeek ผ่าน API (`Script/Generate/generate_datasets_deepseek.py`)
    *   รองรับงาน: Text Classification, Question Answering (QA), Table QA, Zero-Shot Classification, Named Entity Recognition (NER), Translation (TH-EN), Summarization, Sentence Similarity, Text Generation, Style Transfer (Formal/Informal), Fill-Mask, Text Ranking.
*   **Dataset Utilities & Demonstrations (`Script/Dataset/`):**
    *   **Content Moderation:** ตรวจสอบเนื้อหาที่ไม่เหมาะสม (`content_moderation.py`)
    *   **Conversation Simulation:** จำลองและวิเคราะห์บทสนทนา (`conversation_simulation.py`)
    *   **FAQ Summarization:** สรุปคำถามที่พบบ่อยและสร้างคำตอบ (`faq_summarization.py`)
    *   **Named Entity Recognition (NER):** ระบุชื่อเฉพาะในข้อความ (`ner_handler.py`)
    *   **Paraphrase Identification:** ตรวจจับและสร้างประโยคที่มีความหมายเหมือนกัน (`paraphrase_identification.py`)
    *   **Style Transfer:** แปลงรูปแบบภาษาทางการ/ไม่ทางการ (`style_transfer.py`)
    *   **Transcript Analysis:** วิเคราะห์บทสนทนา/เสียงที่ถอดความ (`transcript_handler.py`)
    *   **TTS Script Analysis:** วิเคราะห์สคริปต์สำหรับ Text-to-Speech (`tts_script_handler.py`)
*   **Model Management:**
    *   ดาวน์โหลดโมเดล Pre-trained จาก Hugging Face (`Script/download_model.py`)
    *   อัปโหลดโมเดลไปยัง Hugging Face Hub (`Script/upload_model_to_hf.py`)
*   **Pre-defined Datasets:**
    *   ตัวอย่างชุดข้อมูลในรูปแบบต่างๆ (`Dataset/`, `DatasetCook/`, `DatasetNLP/`, `DatasetReasoning/`)

## 🚀 การติดตั้ง (Installation)

1.  **Clone repository:**
    ```bash
    git clone <your-repository-url>
    cd aibuilderPart1
    ```
2.  **สร้างและเปิดใช้งาน Virtual Environment (แนะนำ):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **ติดตั้ง Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(หมายเหตุ: หากยังไม่มีไฟล์ `requirements.txt` คุณอาจต้องสร้างขึ้นโดย `pip freeze > requirements.txt` หลังจากติดตั้งไลบรารีที่จำเป็น เช่น `transformers`, `torch`, `pandas`, `requests`, `langchain`, `huggingface_hub`)*

4.  **ตั้งค่า Environment Variables (สำหรับ Dataset Generation):**
    *   สร้างไฟล์ `.env` ในไดเรกทอรีรากของโปรเจกต์
    *   เพิ่ม API Key ของคุณ:
      ```dotenv
      DEEPSEEK_API_KEY="your_deepseek_api_key_here"
      ```
    *   สคริปต์ `generate_datasets_deepseek.py` จะโหลดค่านี้โดยอัตโนมัติ (ต้องติดตั้ง `python-dotenv`: `pip install python-dotenv`)

## 🛠️ การใช้งาน (Usage)

ดูรายละเอียดการใช้งานสคริปต์แต่ละตัวฉบับเต็มได้ที่ [**เอกสารการใช้งาน (./docs/USAGE.md)**](./docs/USAGE.md)

**ตัวอย่างการรันเบื้องต้น:**

1.  **ดาวน์โหลดโมเดลพื้นฐาน:**
    ```bash
    python Script/download_model.py
    ```
2.  **รันสคริปต์สร้าง Dataset (ตัวอย่าง):**
    ```bash
    python Script/Generate/generate_datasets_deepseek.py
    ```
    *(ตรวจสอบให้แน่ใจว่าตั้งค่า `DEEPSEEK_API_KEY` ใน `.env` แล้ว)*

3.  **รันสคริปต์สาธิต (ตัวอย่าง - NER):**
    ```bash
    python Script/Dataset/ner_handler.py
    ```
    *(สคริปต์ใน `Script/Dataset/` ส่วนใหญ่จะมีการสาธิตและโหมดโต้ตอบเมื่อรันโดยตรง)*

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
aibuilderPart1/
├── Dataset/              # ตัวอย่างชุดข้อมูล (รูปแบบต่างๆ)
├── DatasetCook/          # ชุดข้อมูลเพิ่มเติม (ตามหัวข้อ)
├── DatasetNLP/           # ชุดข้อมูล NLP เฉพาะทาง
├── DatasetReasoning/     # ชุดข้อมูลสำหรับงาน Reasoning
├── docs/                 # เอกสารประกอบ
│   └── USAGE.md
├── Model/                # ที่เก็บโมเดลที่ดาวน์โหลด/ฝึก
├── Script/               # สคริปต์หลัก
│   ├── Dataset/          # สคริปต์สาธิต/ประเมินผลตามประเภทงาน
│   ├── Generate/         # สคริปต์สร้างชุดข้อมูล
│   ├── download_model.py
│   └── upload_model_to_hf.py
├── DataOutput/           # โฟลเดอร์ผลลัพธ์ (สร้างอัตโนมัติ)
├── README.md             # ไฟล์นี้
├── requirements.txt      # (ควรสร้าง) รายการ Dependencies
└── .env                  # (ต้องสร้าง) เก็บค่า Environment Variables
```

## 🤝 การมีส่วนร่วม (Contributing)

ยินดีรับการมีส่วนร่วม! หากคุณต้องการพัฒนาโปรเจกต์นี้ โปรดพิจารณาขั้นตอนต่อไปนี้:

1.  Fork โปรเจกต์
2.  สร้าง Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit การเปลี่ยนแปลงของคุณ (`git commit -m 'Add some AmazingFeature'`)
4.  Push ไปยัง Branch (`git push origin feature/AmazingFeature`)
5.  เปิด Pull Request

## 📄 สัญญาอนุญาต (License)

โปรเจกต์นี้อยู่ภายใต้สัญญาอนุญาต MIT ดูรายละเอียดเพิ่มเติมได้ที่ไฟล์ [LICENSE](LICENSE)

---

_พัฒนาโดย: [JonusNattapong/zombitx64]_