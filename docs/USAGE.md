# เอกสารการใช้งาน (Usage Documentation)

เอกสารนี้อธิบายวิธีการใช้งานสคริปต์ต่างๆ ในโปรเจกต์ AI Builder Part 1

## สารบัญ

1.  [ข้อกำหนดเบื้องต้น (Prerequisites)](#ข้อกำหนดเบื้องต้น-prerequisites)
2.  [การจัดการโมเดล (Model Management)](#การจัดการโมเดล-model-management)
    *   [ดาวน์โหลดโมเดล](#ดาวน์โหลดโมเดล-scriptdownload_modelpy)
    *   [อัปโหลดโมเดล](#อัปโหลดโมเดล-scriptupload_model_to_hfpy)
3.  [การสร้างชุดข้อมูล (Dataset Generation)](#การสร้างชุดข้อมูล-dataset-generation)
    *   [ใช้ DeepSeek API](#ใช้-deepseek-api-scriptgenerategenerate_datasets_deepseekpy)
    *   [ใช้ LangChain](#ใช้-langchain-scriptgenerategenerate_datasets_langchainpy)
4.  [การใช้งานสคริปต์สาธิต (`Script/Dataset/`)](#การใช้งานสคริปต์สาธิต-scriptdataset)
    *   [Content Moderation](#content-moderation-scriptdatasetcontent_moderationpy)
    *   [Conversation Simulation](#conversation-simulation-scriptdatasetconversation_simulationpy)
    *   [FAQ Summarization](#faq-summarization-scriptdatasetfaq_summarizationpy)
    *   [NER Handler](#ner-handler-scriptdatasetner_handlerpy)
    *   [Paraphrase Identification](#paraphrase-identification-scriptdatasetparaphrase_identificationpy)
    *   [Style Transfer](#style-transfer-scriptdatasetstyle_transferpy)
    *   [Transcript Handler](#transcript-handler-scriptdatasettranscript_handlerpy)
    *   [TTS Script Handler](#tts-script-handler-scriptdatasettts_script_handlerpy)
5.  [ผลลัพธ์ (Outputs)](#ผลลัพธ์-outputs)

---

## ข้อกำหนดเบื้องต้น (Prerequisites)

*   ติดตั้ง Python (เวอร์ชั่น 3.8 ขึ้นไปแนะนำ)
*   ติดตั้ง Dependencies ทั้งหมดตามที่ระบุใน `README.md` (ส่วน Installation)
*   สำหรับ **Dataset Generation**: ต้องมี DeepSeek API Key และตั้งค่าในไฟล์ `.env` (ดู [README.md](./../README.md#การติดตั้ง-installation))

---

## การจัดการโมเดล (Model Management)

### ดาวน์โหลดโมเดล (`Script/download_model.py`)

สคริปต์นี้ใช้สำหรับดาวน์โหลดโมเดล Pre-trained และ Tokenizer จาก Hugging Face Hub มาเก็บไว้ในเครื่อง

*   **การใช้งาน:**
    ```bash
    python Script/download_model.py
    ```
*   **การทำงาน:**
    *   ดาวน์โหลดโมเดลที่ระบุในตัวแปร `MODEL_NAME` (ค่าเริ่มต้น: `airesearch/wangchanberta-base-att-spm-uncased`)
    *   บันทึกไฟล์โมเดลและ Tokenizer ลงในโฟลเดอร์ `Model/<model_name>/` (เช่น `Model/wangchanberta-base-att-spm-uncased/`)
    *   สร้างโฟลเดอร์ `Model` หากยังไม่มี

### อัปโหลดโมเดล (`Script/upload_model_to_hf.py`)

สคริปต์นี้ใช้สำหรับอัปโหลดโฟลเดอร์โมเดลในเครื่องไปยัง Repository บน Hugging Face Hub

*   **การใช้งาน (Command Line Arguments):**
    ```bash
    python Script/upload_model_to_hf.py --local_model_name <ชื่อโฟลเดอร์โมเดลใน Model/> --repo_id <YourUsername/YourRepoName> [--commit_message "Your commit message"] [--create_repo]
    ```
*   **Arguments:**
    *   `--local_model_name` (จำเป็น): ชื่อของโฟลเดอร์ย่อยภายใน `Model/` ที่ต้องการอัปโหลด (เช่น `wangchanberta-base-att-spm-uncased`)
    *   `--repo_id` (จำเป็น): ชื่อ Repository บน Hugging Face Hub (รูปแบบ: `YourUsername/YourRepoName` หรือ `OrgName/RepoName`) **สำคัญ:** ต้องแทนที่ `YourUsername` ด้วยชื่อผู้ใช้ของคุณจริงๆ
    *   `--commit_message` (ทางเลือก): ข้อความสำหรับ Commit (ค่าเริ่มต้น: "Upload model from script")
    *   `--create_repo` (ทางเลือก): Flag เพื่อพยายามสร้าง Repository บน Hub หากยังไม่มี (อาจต้อง Login ผ่าน `huggingface-cli login` ก่อน)
*   **ข้อควรระวัง:** ตรวจสอบให้แน่ใจว่าคุณได้ Login เข้าสู่ Hugging Face Hub ผ่าน CLI (`huggingface-cli login`) ก่อนใช้งานสคริปต์นี้

---

## การสร้างชุดข้อมูล (Dataset Generation)

### ใช้ DeepSeek API (`Script/Generate/generate_datasets_deepseek.py`)

สคริปต์นี้ใช้ DeepSeek API เพื่อสร้างชุดข้อมูลสำหรับงาน NLP ต่างๆ ตามที่กำหนดค่าไว้ใน `Script/Generate/config_generate.py`

*   **การเตรียมการ:**
    *   ตรวจสอบว่ามีไฟล์ `.env` และกำหนด `DEEPSEEK_API_KEY` ถูกต้อง
    *   ติดตั้ง `python-dotenv`: `pip install python-dotenv`
*   **การใช้งาน:**
    ```bash
    python Script/Generate/generate_datasets_deepseek.py
    ```
*   **การทำงาน:**
    *   อ่านค่า Configuration จาก `config_generate.py` (เช่น หัวข้อ, จำนวนตัวอย่าง, ชื่อไฟล์ผลลัพธ์)
    *   วนลูปสร้างข้อมูลสำหรับแต่ละประเภทงาน (Classification, QA, NER, Summarization, Translation, Similarity, Text Generation, Style Transfer, Fill-Mask, Text Ranking, **Code Generation**, **Reasoning (CoT)**) โดยเรียกใช้ DeepSeek API
    *   ใช้ Prompt Templates ที่กำหนดไว้ในสคริปต์ (`generate_datasets_deepseek.py`)
    *   Parse ผลลัพธ์ JSON ที่ได้จาก API
    *   บันทึกข้อมูลที่สร้างได้ลงในไฟล์ CSV ภายในโฟลเดอร์ `DataOutput/` (ชื่อไฟล์ตามที่กำหนดใน `config_generate.py`)
    *   มีระบบ Retry หาก API Request ล้มเหลว

### ใช้ LangChain (`Script/Generate/generate_datasets_langchain.py`)

สคริปต์นี้ใช้ LangChain และ Hugging Face Endpoint (หรือ LLM อื่นๆ ที่รองรับ) เพื่อสร้างชุดข้อมูลสำหรับงาน NLP ต่างๆ ตามที่กำหนดค่าไว้ใน `Script/Generate/config_generate.py` และสคริปต์ย่อย (`gen_*.py`)

*   **การเตรียมการ:**
    *   ตรวจสอบว่ามีไฟล์ `.env` และกำหนด `HUGGINGFACEHUB_API_TOKEN` ถูกต้อง (หากใช้ HuggingFaceEndpoint)
    *   ติดตั้ง `langchain`, `langchain-huggingface` (หรือ library ที่เกี่ยวข้องกับ LLM ที่เลือก)
*   **การใช้งาน:**
    ```bash
    python Script/Generate/generate_datasets_langchain.py
    ```
*   **การทำงาน:**
    *   อ่านค่า Configuration จาก `config_generate.py`
    *   ตั้งค่า LLM ผ่าน LangChain (ตัวอย่างใช้ `HuggingFaceEndpoint`)
    *   เรียกใช้ฟังก์ชัน `generate_<task>` จากสคริปต์ย่อย (`gen_qa.py`, `gen_ner.py`, `gen_code_generation.py`, `gen_reasoning_cot.py`, etc.) สำหรับแต่ละประเภทงาน
    *   แต่ละฟังก์ชันย่อยจะใช้ Prompt Template เฉพาะ, เรียก LLM, Parse ผลลัพธ์, และบันทึกเป็น CSV ใน `DataOutput/`
    *   รองรับงาน: Classification, QA, Table QA, Zero-Shot, NER, Translation, Summarization, Similarity, Text Generation, Style Transfer, Fill-Mask, Text Ranking, **Code Generation**, **Reasoning (CoT)**
    *   มีระบบ Retry หาก API Request ล้มเหลว (ใน `gen_utils.py`)

---

## การใช้งานสคริปต์สาธิต (`Script/Dataset/`)

สคริปต์ในโฟลเดอร์นี้มีไว้เพื่อสาธิตการใช้งานโมเดลพื้นฐาน (`wangchanberta`) สำหรับงาน NLP ต่างๆ โดยใช้ชุดข้อมูลตัวอย่างที่กำหนดไว้ในโค้ด (เช่น `Dataset/ner_dataset.py`, `Dataset/paraphrase_identification_dataset.py`)

**รูปแบบการใช้งานทั่วไป:**

สคริปต์ส่วนใหญ่ใน `Script/Dataset/` สามารถรันได้โดยตรงจาก Command Line:

```bash
python Script/Dataset/<script_name>.py
```

เมื่อรัน สคริปต์มักจะทำงานดังนี้:

1.  **โหลดโมเดล:** โหลดโมเดล `wangchanberta` (หรือโมเดลที่ระบุในโค้ด) และ Tokenizer
2.  **ประเมินผล (Evaluate):** ประมวลผลชุดข้อมูลตัวอย่าง (`evaluate_model()` หรือฟังก์ชันคล้ายกัน) และแสดงผลลัพธ์การทำนายเทียบกับค่าที่คาดหวัง รวมถึงค่า Accuracy (ถ้ามี)
3.  **สาธิต (Demonstrate):** แสดงตัวอย่างการทำงานของโมเดลกับข้อมูลบางส่วน (`demonstrate_usage()`)
4.  **โหมดโต้ตอบ (Interactive Demo):** บางสคริปต์จะมีโหมดให้ผู้ใช้ป้อนข้อมูลและดูผลลัพธ์แบบ Real-time (`run_interactive_demo()`) พิมพ์ 'exit' หรือทำตามคำแนะนำเพื่อออก
5.  **วิเคราะห์รูปแบบ (Analyze Patterns):** บางสคริปต์อาจมีการวิเคราะห์รูปแบบเฉพาะของงานนั้นๆ (`analyze_..._patterns()`)

**สคริปต์ที่น่าสนใจ:**

### Content Moderation (`Script/Dataset/content_moderation.py`)

*   **หน้าที่:** ตรวจจับข้อความว่าเหมาะสมหรือไม่ (Label 0: เหมาะสม, Label 1: ไม่เหมาะสม)
*   **การรัน:** `python Script/Dataset/content_moderation.py`
*   **ผลลัพธ์:** แสดงผลการทำนาย, ความมั่นใจ, และเปรียบเทียบกับ Label ที่ถูกต้อง มีโหมด Interactive

### Conversation Simulation (`Script/Dataset/conversation_simulation.py`)

*   **หน้าที่:** จำลองบทสนทนา, วิเคราะห์ประเภท, อารมณ์, และสร้างการตอบกลับ
*   **การรัน:** `python Script/Dataset/conversation_simulation.py`
*   **ผลลัพธ์:** แสดงการวิเคราะห์บทสนทนาตัวอย่าง, การสร้าง Response, และมีโหมด Interactive

### FAQ Summarization (`Script/Dataset/faq_summarization.py`)

*   **หน้าที่:** สร้างคำตอบสำหรับคำถามจากเอกสาร FAQ และสรุปประเด็นสำคัญ
*   **การรัน:** `python Script/Dataset/faq_summarization.py`
*   **ผลลัพธ์:** แสดงคำตอบที่สร้างขึ้นเทียบกับคำตอบตัวอย่าง และแสดง Key Points ที่สกัดได้ มีโหมด Interactive

### NER Handler (`Script/Dataset/ner_handler.py`)

*   **หน้าที่:** ระบุ Entities (เช่น บุคคล, องค์กร, สถานที่) ในข้อความ
*   **การรัน:** `python Script/Dataset/ner_handler.py`
*   **ผลลัพธ์:** แสดง Entities ที่ตรวจพบพร้อมประเภทและความมั่นใจ คำนวณ Accuracy และมีโหมด Interactive

### Paraphrase Identification (`Script/Dataset/paraphrase_identification.py`)

*   **หน้าที่:** ตรวจสอบว่าประโยคสองประโยคมีความหมายเหมือนกันหรือไม่ และสร้างประโยคใหม่ที่มีความหมายคล้ายกัน
*   **การรัน:** `python Script/Dataset/paraphrase_identification.py`
*   **ผลลัพธ์:** แสดงผลการเปรียบเทียบ, คะแนนความคล้ายคลึง, การวิเคราะห์ความแตกต่าง, และตัวอย่างประโยคที่สร้างขึ้นใหม่ มีโหมด Interactive

### Style Transfer (`Script/Dataset/style_transfer.py`)

*   **หน้าที่:** แปลงรูปแบบภาษาระหว่างทางการ (Formal) และไม่ทางการ (Informal)
*   **การรัน:** `python Script/Dataset/style_transfer.py`
*   **ผลลัพธ์:** แสดงผลการแปลงทั้งสองทิศทาง (Formal -> Informal, Informal -> Formal) และการวิเคราะห์รูปแบบ มีโหมด Interactive

### Transcript Handler (`Script/Dataset/transcript_handler.py`)

*   **หน้าที่:** วิเคราะห์บทสนทนาหรือข้อความที่ถอดความจากเสียง ระบุสภาพแวดล้อม, ผู้พูด, รูปแบบการพูด, และทำความสะอาดข้อความ
*   **การรัน:** `python Script/Dataset/transcript_handler.py`
*   **ผลลัพธ์:** แสดงผลการวิเคราะห์ต่างๆ เช่น สภาพแวดล้อม, ผู้พูด, รูปแบบคำพูด, ความเป็นทางการ มีโหมด Interactive

### TTS Script Handler (`Script/Dataset/tts_script_handler.py`)

*   **หน้าที่:** วิเคราะห์สคริปต์สำหรับ Text-to-Speech เพื่อหาอารมณ์, คุณสมบัติ, และให้คำแนะนำในการสังเคราะห์เสียง
*   **การรัน:** `python Script/Dataset/tts_script_handler.py`
*   **ผลลัพธ์:** แสดงผลการวิเคราะห์อารมณ์, คุณสมบัติข้อความ, และคำแนะนำสำหรับ TTS มีโหมด Interactive

---

## ผลลัพธ์ (Outputs)

*   **โมเดลที่ดาวน์โหลด:** จะถูกเก็บไว้ในโฟลเดอร์ `Model/`
*   **ชุดข้อมูลที่สร้าง:** ไฟล์ CSV จะถูกบันทึกในโฟลเดอร์ `DataOutput/` (สร้างขึ้นอัตโนมัติหากยังไม่มี)
*   **ผลลัพธ์จากสคริปต์สาธิต:** ส่วนใหญ่จะแสดงผลทาง Console โดยตรง บางสคริปต์อาจมีการบันทึกผลลงไฟล์ใน `DataOutput/` (ตรวจสอบโค้ดของแต่ละสคริปต์)

---

หากพบปัญหาหรือมีคำถามเพิ่มเติม โปรดเปิด Issue ใน GitHub repository ของโปรเจกต์
