# AI Builder Part 1: Thai NLP & Vision Dataset Generation and Utilities

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)

โปรเจกต์นี้รวบรวมเครื่องมือและสคริปต์สำหรับการสร้างชุดข้อมูล (Dataset) ภาษาไทยสำหรับงาน Natural Language Processing (NLP) และ Computer Vision ต่างๆ รวมถึงสคริปต์สาธิตการใช้งานโมเดลพื้นฐานสำหรับงานเหล่านั้น

## ✨ คุณสมบัติหลัก (Key Features)

* **NLP Dataset Generation:**
  * ใช้ Large Language Models (LLMs) ผ่าน DeepSeek API หรือ LangChain เพื่อสร้างข้อมูลตัวอย่างสำหรับงาน NLP หลากหลายประเภท:
    * Text Classification
    * Question Answering (Extractive & Abstractive)
    * Table Question Answering
    * Zero-Shot Text Classification
    * Named Entity Recognition (NER)
    * Translation (Thai-English)
    * Summarization
    * Sentence Similarity / Paraphrase Identification
    * Text Generation
    * Style Transfer (Formal/Informal)
    * Fill-Mask
    * Text Ranking
    * Code Generation (Python)
    * Reasoning (Chain-of-Thought)
  * กำหนดค่าได้ง่ายผ่าน `config_generate.py` (หัวข้อ, จำนวนตัวอย่าง)
  * บันทึกผลลัพธ์เป็นไฟล์ CSV ใน `DataOutput/`
* **Vision Dataset Generation (`DatasetVision/`):**
  * ใช้ Hugging Face Inference API เพื่อสร้างข้อมูลตัวอย่างสำหรับงาน Computer Vision:
    * Image Classification
    * Object Detection
    * Text-to-Image
    * Depth Estimation
    * Image Segmentation
    * Image-to-Text (Captioning)
    * Zero-Shot Image Classification
    * *(Placeholder scripts for other tasks like Text-to-Video, Image-to-Video, etc.)*
  * **สร้าง CSV จากรูปภาพ:** สแกนไดเรกทอรีและสร้างไฟล์ CSV ที่ระบุตำแหน่งรูปภาพ (`create_image_dataset_csv.py`)
    * สามารถใช้ชื่อโฟลเดอร์ย่อยเป็น Label ได้โดยอัตโนมัติ
    * **(ใหม่)** สามารถใช้โมเดล Image Classification ในเครื่อง (Local Hugging Face model) เพื่อสร้าง Label ให้กับรูปภาพได้ (ต้องติดตั้ง `transformers` และ `torch`)
  * กำหนดค่าได้ง่ายผ่าน `config_vision.py` (Model IDs, Input Data, Output Paths)
  * บันทึกผลลัพธ์เป็นไฟล์ CSV ใน `DataOutput/` และเก็บไฟล์ Media ที่สร้างขึ้นใน `DataOutput/generated_media/`
* **(ใหม่) Audio Dataset Generation (`DatasetAudio/`):**
  * ใช้ Hugging Face Inference API (หรือ Local Models ถ้าติดตั้ง) เพื่อสร้างข้อมูลตัวอย่างสำหรับงานด้านเสียง:
    * Text-to-Speech (TTS): สร้างไฟล์เสียงจากข้อความ
    * Automatic Speech Recognition (ASR): แปลงไฟล์เสียงเป็นข้อความ
    * Audio Classification: จำแนกประเภทเสียง (เช่น เสียงดนตรี, เสียงพูด, เสียงสัตว์)
    * Text-to-Audio (Sound Generation): สร้างเสียงตามคำอธิบาย (ทดลอง)
    * Voice Activity Detection (VAD): ตรวจจับช่วงเวลาที่มีเสียงพูดในไฟล์เสียง (ทดลอง)
    * Audio-to-Audio: แปลงเสียงรูปแบบหนึ่งไปอีกรูปแบบหนึ่ง (เช่น ลดเสียงรบกวน) (Placeholder/ทดลอง)
  * **สร้าง CSV จากไฟล์เสียง:** สแกนไดเรกทอรีและสร้างไฟล์ CSV ที่ระบุตำแหน่งไฟล์เสียง (`create_audio_dataset_csv.py`)
    * สามารถใช้ชื่อโฟลเดอร์ย่อยเป็น Label ได้โดยอัตโนมัติ
  * กำหนดค่าได้ง่ายผ่าน `config_audio.py` (Model IDs, Input Data, Output Paths)
  * บันทึกผลลัพธ์เป็นไฟล์ CSV ใน `DataOutput/` และเก็บไฟล์เสียงที่สร้างขึ้นใน `DataOutput/generated_media/audio/`
* **(ใหม่) Multimodal Dataset Generation (`DatasetMultimodal/`):**
  * ใช้ Local Hugging Face Models (ต้องติดตั้ง `transformers`, `torch`, `Pillow`, `decord`) เพื่อสร้างข้อมูลตัวอย่างสำหรับงาน Multimodal:
    * Visual Question Answering (VQA): สร้างคำตอบสำหรับคำถามเกี่ยวกับรูปภาพ
    * Video-Text-to-Text (Video Captioning): สร้างคำบรรยายสำหรับวิดีโอ
  * กำหนดค่าได้ง่ายผ่าน `config_multimodal.py` (Model IDs, Input Paths, Output Paths)
  * บันทึกผลลัพธ์เป็นไฟล์ CSV ใน `DataOutput/`
* **(ใหม่) Tabular Dataset Generation (`DatasetTabular/`):**
  * สร้างข้อมูลตัวอย่างสำหรับงานเกี่ยวกับข้อมูลตาราง:
    * Tabular Classification (Simulated): สร้างข้อมูลตารางพร้อม Label สำหรับ Classification
    * Tabular Regression (Simulated): สร้างข้อมูลตารางพร้อม Target Value สำหรับ Regression
    * Tabular-to-Text (Local T5 Model): สร้างคำอธิบายข้อความจากตารางที่จำลองขึ้น โดยใช้โมเดล T5 ในเครื่อง (ต้องติดตั้ง `transformers`, `torch`)
    * Time Series Forecasting (Simulated): สร้างข้อมูลอนุกรมเวลาแบบง่ายๆ
  * กำหนดค่าได้ง่ายผ่าน `config_tabular.py` (จำนวนตัวอย่าง, ขนาดข้อมูล, Model ID สำหรับ T5)
  * บันทึกผลลัพธ์เป็นไฟล์ CSV ใน `DataOutput/`
* **Dataset Utilities & Demonstrations (`Script/Dataset/`):**
  * **Content Moderation:** ตรวจสอบเนื้อหาที่ไม่เหมาะสม (`content_moderation.py`)
  * **Conversation Simulation:** จำลองและวิเคราะห์บทสนทนา (`conversation_simulation.py`)
  * **Emotion Detection:** วิเคราะห์อารมณ์ในข้อความ (`emotion_detection.py`)
  * **FAQ Summarization:** สรุปคำถามที่พบบ่อยและสร้างคำตอบ (`faq_summarization.py`)
  * **Named Entity Recognition (NER):** ระบุชื่อเฉพาะในข้อความ (`ner_handler.py`)
  * **Paraphrase Identification:** ตรวจจับและสร้างประโยคที่มีความหมายเหมือนกัน (`paraphrase_identification.py`)
  * **Style Transfer:** แปลงรูปแบบภาษาทางการ/ไม่ทางการ (`style_transfer.py`)
  * **Transcript Analysis:** วิเคราะห์บทสนทนา/เสียงที่ถอดความ (`transcript_handler.py`)
  * **TTS Script Analysis:** วิเคราะห์สคริปต์สำหรับ Text-to-Speech (`tts_script_handler.py`)
* **Model Management:**
  * ดาวน์โหลดโมเดล Pre-trained จาก Hugging Face (`Script/download_model.py`)
  * อัปโหลดโมเดลไปยัง Hugging Face Hub (`Script/upload_model_to_hf.py`)
* **Pre-defined Datasets:**
  * ตัวอย่างชุดข้อมูลในรูปแบบต่างๆ (`Dataset/`, `DatasetCook/`, `DatasetNLP/`, `DatasetReasoning/`)
  * ชุดข้อมูลเนื้อหาที่ไม่เหมาะสม (Uncensored) แบบแยกหมวดหมู่ (`DatasetCook/DatasetUncensore/`) สำหรับการวิจัยระบบกรองเนื้อหา
  * **(ใหม่)** ชุดข้อมูลเฉพาะทาง (Domain-Specific) ในรูปแบบ Parquet (`Dataset/Parquet/`) ครอบคลุมหลาย NLP tasks สำหรับโดเมน:
    * `finance_dataset.py` -> `finance_data.parquet`
    * `legal_dataset.py` -> `legal_data.parquet`
    * `medical_dataset.py` -> `medical_data.parquet`
    * `retail_dataset.py` -> `retail_data.parquet`
    * `code_dataset.py` -> `code_data.parquet`
    * `art_dataset.py` -> `art_data.parquet`
    * `chemistry_dataset.py` -> `chemistry_data.parquet`
    * `biology_dataset.py` -> `biology_data.parquet`
    * `music_dataset.py` -> `music_data.parquet`
    * `climate_dataset.py` -> `climate_data.parquet`
    * *(สามารถรันไฟล์ `.py` เพื่อสร้างไฟล์ `.parquet` ที่เกี่ยวข้องใน `DataOutput/`)*
  * *(Vision/Audio/Multimodal/Tabular datasets are generated into `DataOutput/`)*

## 🚀 การติดตั้ง (Installation)

1. **Clone Repository:**

    ```bash
    git clone https://github.com/JonusNattapong/aibuilderPart1.git
    cd aibuilderPart1
    ```

2. **สร้าง Virtual Environment (แนะนำ):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # บน Linux/macOS
    # หรือ
    venv\Scripts\activate    # บน Windows
    ```

3. **ติดตั้ง Dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install Pillow # Needed for Vision tasks
    # Optional: Install torch and transformers for local Vision model inference
    # pip install torch transformers # Or follow official PyTorch installation guide for GPU support
    ```

    *(หมายเหตุ: หากยังไม่มีไฟล์ `requirements.txt` คุณอาจต้องสร้างขึ้นโดย `pip freeze > requirements.txt` หลังจากติดตั้งไลบรารีที่จำเป็น เช่น `transformers`, `torch`, `pandas`, `requests`, `langchain`, `huggingface_hub`, `python-dotenv`, `Pillow`, `accelerate`)*

4. **ตั้งค่า Environment Variables:**
    * สร้างไฟล์ `.env` ในไดเรกทอรีรากของโปรเจกต์
    * **สำหรับ NLP Dataset Generation (DeepSeek):** เพิ่ม API Key ของคุณ:

      ```dotenv
      DEEPSEEK_API_KEY="your_deepseek_api_key_here"
      ```

      * สคริปต์ `generate_datasets_deepseek.py` จะโหลดค่านี้โดยอัตโนมัติ (ต้องติดตั้ง `python-dotenv`: `pip install python-dotenv`)
    * **สำหรับ Vision Dataset Generation (Hugging Face API):** เพิ่ม API Token ของคุณ:

      ```dotenv
      HF_TOKEN="your_hf_api_token_here"
      ```

      * สคริปต์ใน `DatasetVision/` จะโหลดค่านี้โดยอัตโนมัติผ่าน `vision_utils.py`. คุณสามารถขอ Token ได้จาก [Hugging Face settings](https://huggingface.co/settings/tokens).
    * **สำหรับ NLP Dataset Generation (LangChain with HF Endpoint):** อาจต้องตั้งค่า `HUGGINGFACEHUB_API_TOKEN` ด้วย Token เดียวกับ `HF_TOKEN` หากใช้ `HuggingFaceEndpoint`.

      ```dotenv
      HUGGINGFACEHUB_API_TOKEN="your_hf_api_token_here"
      ```

## 🛠️ การใช้งาน (Usage)

ดูรายละเอียดการใช้งานสคริปต์แต่ละตัวฉบับเต็มได้ที่ [**เอกสารการใช้งาน (./docs/USAGE.md)**](./docs/USAGE.md)

**ตัวอย่างการรันเบื้องต้น:**

1. **ดาวน์โหลดโมเดลพื้นฐาน (สำหรับสคริปต์สาธิต NLP):**

    ```bash
    python Script/download_model.py
    ```

2. **รันสคริปต์สร้าง NLP Dataset (ตัวอย่าง - DeepSeek):**

    ```bash
    python Script/Generate/generate_datasets_deepseek.py
    ```

    *(ตรวจสอบให้แน่ใจว่าตั้งค่า `DEEPSEEK_API_KEY` ใน `.env` แล้ว)*

3. **รันสคริปต์สร้าง Vision Dataset (ตัวอย่าง - Text-to-Image):**

    ```bash
    python DatasetVision/gen_text_to_image.py
    ```

    *(ตรวจสอบให้แน่ใจว่าตั้งค่า `HF_TOKEN` ใน `.env` แล้ว และอาจต้องเตรียม Input Images ใน `placeholder_images/` สำหรับบางสคริปต์)*

4. **รันสคริปต์สาธิต (ตัวอย่าง - NER):**

    ```bash
    python Script/Dataset/ner_handler.py
    ```

    *(สคริปต์ใน `Script/Dataset/` ส่วนใหญ่จะมีการสาธิตและโหมดโต้ตอบเมื่อรันโดยตรง)*

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
aibuilderPart1/
├── Dataset/              # ตัวอย่างชุดข้อมูล NLP (รูปแบบต่างๆ)
├── DatasetCook/          # ชุดข้อมูล NLP เพิ่มเติม (ตามหัวข้อ)
├── DatasetNLP/           # ชุดข้อมูล NLP เฉพาะทาง
├── DatasetReasoning/     # ชุดข้อมูล NLP สำหรับงาน Reasoning
├── DatasetVision/        # สคริปต์สร้างชุดข้อมูล Vision
│   ├── config_vision.py
│   ├── vision_utils.py
│   ├── create_image_dataset_csv.py
│   ├── gen_*.py
│   └── ...
├── DatasetAudio/         # (ใหม่) สคริปต์สร้างชุดข้อมูล Audio
│   ├── config_audio.py
│   ├── audio_utils.py
│   ├── create_audio_dataset_csv.py
│   ├── gen_*.py
│   └── ...
├── DatasetMultimodal/    # (ใหม่) สคริปต์สร้างชุดข้อมูล Multimodal (Local Models)
│   ├── config_multimodal.py
│   ├── multimodal_utils.py
│   ├── gen_*.py
│   └── ...
├── DatasetTabular/       # (ใหม่) สคริปต์สร้างชุดข้อมูล Tabular (Simulated/Local)
│   ├── config_tabular.py
│   ├── tabular_utils.py
│   ├── gen_tabular_classification.py
│   ├── gen_tabular_regression.py
│   ├── gen_tabular_to_text.py
│   ├── gen_time_series_forecasting.py
│   └── ...
├── docs/                 # เอกสารประกอบ
│   └── USAGE.md
├── Model/                # ที่เก็บโมเดลที่ดาวน์โหลด/ฝึก (สำหรับ NLP demos)
├── Script/               # สคริปต์หลัก (NLP)
│   ├── Dataset/
│   ├── Generate/
│   ├── download_model.py
│   └── upload_model_to_hf.py
├── placeholder_images/   # (แนะนำให้สร้าง) ที่เก็บรูปภาพ Input สำหรับ Vision/Multimodal tasks
├── placeholder_audio/    # (แนะนำให้สร้าง) ที่เก็บไฟล์เสียง Input สำหรับ Audio tasks
├── placeholder_videos/   # (แนะนำให้สร้าง) ที่เก็บไฟล์วิดีโอ Input สำหรับ Multimodal tasks
├── DataOutput/           # โฟลเดอร์ผลลัพธ์ (สร้างอัตโนมัติ)
│   ├── generated_media/
│   │   ├── audio/
│   │   └── ...
│   └── *.csv
├── README.md             # ไฟล์นี้
├── requirements.txt      # รายการ Dependencies
└── .env                  # (ต้องสร้าง) เก็บค่า Environment Variables
```

## 🤝 การมีส่วนร่วม (Contributing)

ยินดีรับการมีส่วนร่วม! หากคุณต้องการพัฒนาโปรเจกต์นี้ โปรดพิจารณาขั้นตอนต่อไปนี้:

1. Fork โปรเจกต์
2. สร้าง Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลงของคุณ (`git commit -m 'Add some AmazingFeature'`)
4. Push ไปยัง Branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

## 📄 สัญญาอนุญาต (License)

โปรเจกต์นี้อยู่ภายใต้สัญญาอนุญาต MIT ดูรายละเอียดเพิ่มเติมได้ที่ไฟล์ [LICENSE](LICENSE)

---

*พัฒนาโดย: [JonusNattapong/zombitx64]*
