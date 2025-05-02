# เอกสารการใช้งาน (Usage Documentation)

เอกสารนี้อธิบายวิธีการใช้งานสคริปต์ต่างๆ ในโปรเจกต์ AI Builder Part 1

## สารบัญ

1. [ข้อกำหนดเบื้องต้น (Prerequisites)](#ข้อกำหนดเบื้องต้น-prerequisites)
2. [การจัดการโมเดล (Model Management)](#การจัดการโมเดล-model-management)
    * [ดาวน์โหลดโมเดล](#ดาวน์โหลดโมเดล-scriptdownload_modelpy)
    * [อัปโหลดโมเดล](#อัปโหลดโมเดล-scriptupload_model_to_hfpy)
3. [การสร้างชุดข้อมูล (Dataset Generation)](#การสร้างชุดข้อมูล-dataset-generation)
    * [ใช้ DeepSeek API (NLP)](#ใช้-deepseek-api-nlp-scriptgenerategenerate_datasets_deepseekpy)
    * [ใช้ LangChain (NLP)](#ใช้-langchain-nlp-scriptgenerategenerate_datasets_langchainpy)
    * [ใช้ Hugging Face API (Vision)](#ใช้-hugging-face-api-vision-datasetvision)
    * [การใช้ชุดข้อมูลที่กำหนดไว้แล้ว (`Dataset*/`)](#การใช้ชุดข้อมูลที่กำหนดไว้แล้ว-dataset)
4. [การใช้งานสคริปต์สาธิต (`Script/Dataset/`)](#การใช้งานสคริปต์สาธิต-scriptdataset)
    * [Content Moderation](#content-moderation-scriptdatasetcontent_moderationpy)
    * [Conversation Simulation](#conversation-simulation-scriptdatasetconversation_simulationpy)
    * [Emotion Detection](#emotion-detection-scriptdatasetemotion_detectionpy)
    * [FAQ Summarization](#faq-summarization-scriptdatasetfaq_summarizationpy)
    * [NER Handler](#ner-handler-scriptdatasetner_handlerpy)
    * [Paraphrase Identification](#paraphrase-identification-scriptdatasetparaphrase_identificationpy)
    * [Style Transfer](#style-transfer-scriptdatasetstyle_transferpy)
    * [Transcript Handler](#transcript-handler-scriptdatasettranscript_handlerpy)
    * [TTS Script Handler](#tts-script-handler-scriptdatasettts_script_handlerpy)
5. [ผลลัพธ์ (Outputs)](#ผลลัพธ์-outputs)

---

## ข้อกำหนดเบื้องต้น (Prerequisites)

* ติดตั้ง Python (เวอร์ชั่น 3.8 ขึ้นไปแนะนำ)
* ติดตั้ง Dependencies ทั้งหมดตามที่ระบุใน `README.md` (ส่วน Installation), รวมถึง:
  * `transformers`, `torch`, `pandas`, `requests`, `langchain`, `huggingface_hub`, `python-dotenv`
  * `Pillow` (สำหรับ Vision tasks)
* สำหรับ **NLP Dataset Generation (DeepSeek)**: ต้องมี DeepSeek API Key และตั้งค่าในไฟล์ `.env` (ดู [README.md](./../README.md#การติดตั้ง-installation))
* สำหรับ **Vision Dataset Generation (Hugging Face API)**: ต้องมี Hugging Face User Access Token (ที่มีสิทธิ์ `read` หรือ `write` ขึ้นอยู่กับโมเดล) และตั้งค่าในไฟล์ `.env` เป็น `HF_TOKEN` (ดู [README.md](./../README.md#การติดตั้ง-installation))
* สำหรับ **NLP Dataset Generation (LangChain with HF Endpoint)**: อาจต้องตั้งค่า `HUGGINGFACEHUB_API_TOKEN` ใน `.env` ด้วย Token เดียวกับ `HF_TOKEN`.
* สำหรับ **Model Upload**: ต้อง Login เข้า Hugging Face Hub ผ่าน CLI (`huggingface-cli login`)

---

## การจัดการโมเดล (Model Management)

### ดาวน์โหลดโมเดล (`Script/download_model.py`)

สคริปต์นี้ใช้สำหรับดาวน์โหลดโมเดล Pre-trained และ Tokenizer จาก Hugging Face Hub มาเก็บไว้ในเครื่อง (ส่วนใหญ่ใช้สำหรับสคริปต์สาธิต NLP ใน `Script/Dataset/`)

* **การใช้งาน:**

    ```bash
    python Script/download_model.py
    ```

* **การทำงาน:**
  * ดาวน์โหลดโมเดลที่ระบุในตัวแปร `MODEL_NAME` (ค่าเริ่มต้น: `airesearch/wangchanberta-base-att-spm-uncased`)
  * สร้างโฟลเดอร์ `Model` หากยังไม่มี
  * บันทึกไฟล์โมเดลและ Tokenizer ลงในโฟลเดอร์ `Model/<model_name>/` (เช่น `Model/wangchanberta-base-att-spm-uncased/`)

### อัปโหลดโมเดล (`Script/upload_model_to_hf.py`)

สคริปต์นี้ใช้สำหรับอัปโหลดโฟลเดอร์โมเดลในเครื่องไปยัง Repository บน Hugging Face Hub

* **การใช้งาน (Command Line Arguments):**

    ```bash
    python Script/upload_model_to_hf.py --local_model_name <ชื่อโฟลเดอร์โมเดลใน Model/> --repo_id <YourUsername/YourRepoName> [--commit_message "Your commit message"] [--create_repo]
    ```

* **Arguments:**
  * `--local_model_name` (จำเป็น): ชื่อของโฟลเดอร์ย่อยภายใน `Model/` ที่ต้องการอัปโหลด (เช่น `wangchanberta-base-att-spm-uncased`)
  * `--repo_id` (จำเป็น): ชื่อ Repository บน Hugging Face Hub (รูปแบบ: `YourUsername/YourRepoName` หรือ `OrgName/RepoName`) **สำคัญ:** ต้องแทนที่ `YourUsername` ด้วยชื่อผู้ใช้ของคุณจริงๆ
  * `--commit_message` (ทางเลือก): ข้อความสำหรับ Commit (ค่าเริ่มต้น: "Upload model from script")
  * `--create_repo` (ทางเลือก): Flag เพื่อพยายามสร้าง Repository บน Hub หากยังไม่มี (อาจต้อง Login ผ่าน `huggingface-cli login` ก่อน)
* **ข้อควรระวัง:** ตรวจสอบให้แน่ใจว่าคุณได้ Login เข้าสู่ Hugging Face Hub ผ่าน CLI (`huggingface-cli login`) ก่อนใช้งานสคริปต์นี้

---

## การสร้างชุดข้อมูล (Dataset Generation)

### ใช้ DeepSeek API (NLP) (`Script/Generate/generate_datasets_deepseek.py`)

สคริปต์นี้ใช้ DeepSeek API เพื่อสร้างชุดข้อมูลสำหรับงาน NLP ต่างๆ ตามที่กำหนดค่าไว้ใน `Script/Generate/config_generate.py`

* **การเตรียมการ:**
  * ตรวจสอบว่ามีไฟล์ `.env` และกำหนด `DEEPSEEK_API_KEY` ถูกต้อง
  * ติดตั้ง `python-dotenv`: `pip install python-dotenv`
* **การใช้งาน:**

    ```bash
    python Script/Generate/generate_datasets_deepseek.py
    ```

* **การทำงาน:**
  * อ่านค่า Configuration จาก `config_generate.py` (เช่น หัวข้อ, จำนวนตัวอย่าง, ชื่อไฟล์ผลลัพธ์)
  * วนลูปสร้างข้อมูลสำหรับแต่ละประเภทงาน (Classification, QA, NER, Summarization, Translation, Similarity, Text Generation, Style Transfer, Fill-Mask, Text Ranking, Code Generation, Reasoning (CoT)) โดยเรียกใช้ DeepSeek API
  * ใช้ Prompt Templates ที่กำหนดไว้ในสคริปต์ (`generate_datasets_deepseek.py`)
  * Parse ผลลัพธ์ JSON ที่ได้จาก API
  * บันทึกข้อมูลที่สร้างได้ลงในไฟล์ CSV ภายในโฟลเดอร์ `DataOutput/` (ชื่อไฟล์ตามที่กำหนดใน `config_generate.py`)
  * มีระบบ Retry หาก API Request ล้มเหลว

### ใช้ LangChain (NLP) (`Script/Generate/generate_datasets_langchain.py`)

สคริปต์นี้ใช้ LangChain framework ร่วมกับ Hugging Face Endpoint (หรือ LLM อื่นๆ ที่ LangChain รองรับ) เพื่อสร้างชุดข้อมูล NLP

* **การเตรียมการ:**
  * ตรวจสอบว่าตั้งค่า Environment Variable ที่จำเป็นสำหรับ LLM ที่เลือก (เช่น `HUGGINGFACEHUB_API_TOKEN` สำหรับ `HuggingFaceEndpoint`)
* **การใช้งาน:**

    ```bash
    python Script/Generate/generate_datasets_langchain.py
    ```

* **การทำงาน:**
  * อ่านค่า Configuration จาก `config_generate.py`
  * ตั้งค่า LLM ผ่าน LangChain (ตัวอย่างใช้ `HuggingFaceEndpoint`)
  * เรียกใช้ฟังก์ชัน `generate_<task>` จากสคริปต์ย่อย (`gen_qa.py`, `gen_ner.py`, `gen_code_generation.py`, `gen_reasoning_cot.py`, etc.) สำหรับแต่ละประเภทงาน
  * แต่ละฟังก์ชันย่อยจะใช้ Prompt Template เฉพาะ, เรียก LLM, Parse ผลลัพธ์, และบันทึกเป็น CSV ใน `DataOutput/`
  * รองรับงาน: Classification, QA, Table QA, Zero-Shot, NER, Translation, Summarization, Similarity, Text Generation, Style Transfer, Fill-Mask, Text Ranking, Code Generation, Reasoning (CoT)
  * มีระบบ Retry หาก API Request ล้มเหลว (ใน `gen_utils.py`)

### ใช้ Hugging Face API (Vision) (`DatasetVision/`)

สคริปต์ในโฟลเดอร์ `DatasetVision/` ใช้ Hugging Face Inference API เพื่อสร้างชุดข้อมูลสำหรับงาน Computer Vision ต่างๆ

* **การเตรียมการ:**
  * ตรวจสอบว่ามีไฟล์ `.env` และกำหนด `HF_TOKEN` ถูกต้อง (ดู [README.md](./../README.md#การติดตั้ง-installation))
  * ติดตั้ง Dependencies: `pip install -r requirements.txt` (โดยเฉพาะ `requests`, `Pillow`, `pandas`)
  * (แนะนำ) สร้างโฟลเดอร์ `placeholder_images/` ในระดับรากของโปรเจกต์ และใส่รูปภาพตัวอย่างสำหรับ Input ของสคริปต์ `gen_*.py` ต่างๆ (เช่น สำหรับ Image Classification, Object Detection) หรือแก้ไข `config_vision.py` ให้ชี้ไปยัง Path ที่ถูกต้อง

* **การใช้งาน (ตัวอย่าง):**

    ```bash
    # รันสคริปต์สร้างข้อมูลแต่ละประเภท (ใช้ API)
    python DatasetVision/gen_image_classification.py
    python DatasetVision/gen_text_to_image.py
    python DatasetVision/gen_depth_estimation.py
    # ... และสคริปต์ gen_*.py อื่นๆ ใน DatasetVision/

    # รันสคริปต์สร้าง CSV จากรูปภาพที่มีอยู่
    python DatasetVision/create_image_dataset_csv.py --input_dir path/to/your/images --output_filename my_images.csv [--recursive | --no-recursive]
    ```

* **การทำงาน (สคริปต์ `gen_*.py`):**
  * อ่านค่า Configuration จาก `DatasetVision/config_vision.py` (เช่น Model ID, จำนวนตัวอย่าง, Input Data Paths, ชื่อไฟล์ผลลัพธ์)
  * เรียกใช้ Hugging Face Inference API ผ่าน `vision_utils.py` เพื่อประมวลผล Input (รูปภาพ หรือ Prompt)
  * บันทึกข้อมูล Metadata (เช่น Path รูปภาพ Input/Output, Prompt, Predictions) ลงในไฟล์ CSV ภายในโฟลเดอร์ `DataOutput/` (ชื่อไฟล์ตามที่กำหนดใน `config_vision.py`)
  * สำหรับ Task ที่สร้าง Output เป็นรูปภาพ (เช่น Text-to-Image, Depth Estimation) จะบันทึกไฟล์รูปลงใน `DataOutput/generated_media/<task_name>/`
  * มีระบบ Retry หาก API Request ล้มเหลว (ใน `vision_utils.py`)

* **การทำงาน (สคริปต์ `create_image_dataset_csv.py`):**
  * รับ Path ของไดเรกทอรีที่มีรูปภาพ (`--input_dir`) และชื่อไฟล์ CSV ผลลัพธ์ (`--output_filename`) เป็น Argument (ค่าเริ่มต้น `--input_dir` คือ `DatasetVision/Img`)
  * สแกนหาไฟล์รูปภาพในไดเรกทอรีที่ระบุ (ค่าเริ่มต้นคือสแกนแบบ Recursive `--recursive`, ใช้ `--no-recursive` หากไม่ต้องการ)
  * สร้างไฟล์ CSV ใน `DataOutput/` ที่มีคอลัมน์ `id` (UUID) และ `image_path` (Path สัมพัทธ์จากรากโปรเจกต์)
  * **การกำหนด Label:**
    * **ค่าเริ่มต้น:** หากรูปภาพถูกจัดอยู่ในโฟลเดอร์ย่อย จะใช้ชื่อโฟลเดอร์ย่อยนั้นเป็น `label` ในไฟล์ CSV (ถ้าไม่มีโฟลเดอร์ย่อย หรือสแกนแบบไม่ Recursive ที่ระดับบนสุด Label จะเป็น "unknown")
    * **(ใหม่) ใช้ Local Model:** หากระบุ Argument `--local_classifier_model <model_name_or_path>` สคริปต์จะพยายามโหลดโมเดล Image Classification จาก Hugging Face Hub (หรือ Path ในเครื่อง) โดยใช้ไลบรารี `transformers` และ `torch` เพื่อทำนาย Label ของแต่ละภาพ Label ที่ได้จากโมเดลจะถูกนำมาใช้แทนที่ Label ที่ได้จากชื่อโฟลเดอร์
      * **ข้อกำหนด:** ต้องติดตั้ง `pip install transformers torch` (แนะนำให้ติดตั้ง PyTorch ตาม [คำแนะนำอย่างเป็นทางการ](https://pytorch.org/get-started/locally/) เพื่อรองรับ GPU หากมี)
      * **ตัวอย่าง Model:** `"google/vit-base-patch16-224"`, `"microsoft/beit-base-patch16-224"`, หรือโมเดลอื่นๆ ที่รองรับ `AutoModelForImageClassification`
      * **ตัวอย่างคำสั่ง:**
        ```bash
        # ใช้โมเดล ViT ทำนาย Label จากรูปใน DatasetVision/Img (ต้องติดตั้ง transformers, torch)
        python DatasetVision/create_image_dataset_csv.py --local_classifier_model google/vit-base-patch16-224 --output_filename vision_dataset_with_model_labels.csv

        # ใช้โมเดล ViT ทำนาย Label จากรูปในโฟลเดอร์อื่น
        python DatasetVision/create_image_dataset_csv.py --input_dir path/to/other/images --local_classifier_model google/vit-base-patch16-224 --output_filename other_images_model_labels.csv
        ```

* **สคริปต์ที่รองรับ API (ปัจจุบัน):**
  * `gen_image_classification.py`
  * `gen_object_detection.py`
  * `gen_text_to_image.py`
  * `gen_depth_estimation.py`
  * `gen_image_segmentation.py`
  * `gen_image_to_text.py`
  * `gen_zero_shot_image_classification.py`
* **สคริปต์ Placeholder (ยังไม่ใช้ API):**
  * `gen_image_feature_extraction.py`
  * `gen_image_to_3d.py`
  * `gen_image_to_image.py`
  * `gen_image_to_video.py`
  * `gen_keypoint_detection.py`
  * `gen_mask_generation.py`
  * `gen_text_to_3d.py`
  * `gen_text_to_video.py`
  * `gen_unconditional_image_generation.py`
  * `gen_video_classification.py`
  * `gen_zero_shot_object_detection.py`

### การใช้ชุดข้อมูลที่กำหนดไว้แล้ว (`Dataset*/`)

ในโปรเจคท์นี้มีชุดข้อมูลที่กำหนดไว้แล้วหลายประเภทสำหรับงานต่างๆ:

#### ชุดข้อมูลทั่วไป (`Dataset/`)

ดูไฟล์ในโฟลเดอร์ `Dataset/CSV/` และ `Dataset/Parquet/` สำหรับชุดข้อมูลปกติที่ใช้สำหรับงาน NLP ต่างๆ

#### ชุดข้อมูลตามหัวข้อ (`DatasetCook/`)

โฟลเดอร์ `DatasetCook/` มีไฟล์สคริปต์ Python ที่สร้างชุดข้อมูลแยกตามหัวข้อ เช่น `emotion_dataset.py`, `health_dataset.py` ฯลฯ แต่ละไฟล์รันได้โดยตรงและจะสร้างไฟล์ CSV ใน `DataOutput/`

#### ชุดข้อมูลเนื้อหาที่ไม่เหมาะสม (`DatasetCook/DatasetUncensore/`)

โฟลเดอร์นี้มีชุดข้อมูลสำหรับงานวิจัยด้านการกรองเนื้อหาที่ไม่เหมาะสม (Content Moderation) ซึ่งถูกจัดโครงสร้างแบบแยกส่วน:

* **ไฟล์หลัก:** `uncensored_dataset.py` - ทำหน้าที่รวมข้อมูลจากไฟล์ย่อยทุกไฟล์และสร้างชุดข้อมูลสมบูรณ์
* **ไฟล์ข้อมูลย่อย:** ไฟล์ต่างๆ ที่ลงท้ายด้วย `_data.py` เช่น `offensive_language_data.py`, `adult_content_data.py` ฯลฯ แต่ละไฟล์เก็บข้อมูลเฉพาะประเภทนั้นๆ

**การใช้งาน:**

1. **การรันสคริปต์หลัก:**

   ```bash
   python DatasetCook/DatasetUncensore/uncensored_dataset.py
   ```

   คำสั่งนี้จะรวมข้อมูลจากทุกไฟล์และสร้าง `DataOutput/thai_uncensored_dataset.csv`

2. **การเพิ่มข้อมูลในหมวดหมู่ที่มีอยู่แล้ว:**
   * เปิดไฟล์ของหมวดหมู่นั้น เช่น `offensive_language_data.py`
   * เพิ่มตัวอย่างข้อความลงในลิสต์

3. **การเพิ่มหมวดหมู่ใหม่:**
   * สร้างไฟล์ใหม่ เช่น `new_category_data.py` ที่มีโครงสร้างเหมือนไฟล์อื่นๆ
   * เพิ่ม import และเพิ่มหมวดหมู่ในตัวแปร `categories` ใน `uncensored_dataset.py`

**คำเตือน:** ชุดข้อมูลนี้มีเนื้อหาที่อาจไม่เหมาะสม ใช้สำหรับการวิจัยและพัฒนาระบบกรองเนื้อหาอัตโนมัติเท่านั้น

---

## (ใหม่) การสร้างชุดข้อมูลเสียง (`DatasetAudio/`)

สคริปต์ในโฟลเดอร์ `DatasetAudio/` ใช้สำหรับสร้างชุดข้อมูลสำหรับงานด้านเสียงต่างๆ

### การสร้าง CSV จากไฟล์เสียงที่มีอยู่ (`create_audio_dataset_csv.py`)

สคริปต์นี้ใช้สำหรับสแกนหาไฟล์เสียงในไดเรกทอรีที่ระบุ และสร้างไฟล์ CSV ที่มีรายการไฟล์เสียงเหล่านั้น พร้อม Label ที่ได้จากชื่อโฟลเดอร์ย่อย (ถ้ามี)

*   **การทำงาน:**
    *   รับ Path ของไดเรกทอรีที่มีไฟล์เสียง (`--input_dir`) และชื่อไฟล์ CSV ผลลัพธ์ (`--output_filename`) เป็น Argument (ค่าเริ่มต้น `--input_dir` คือ `placeholder_audio/`)
    *   สแกนหาไฟล์เสียง (เช่น `.wav`, `.mp3`, `.flac`) ในไดเรกทอรีที่ระบุ (ค่าเริ่มต้นคือสแกนแบบ Recursive `--recursive`, ใช้ `--no-recursive` หากไม่ต้องการ)
    *   สร้างไฟล์ CSV ใน `DataOutput/` ที่มีคอลัมน์ `id` (UUID) และ `audio_path` (Path สัมพัทธ์จากรากโปรเจกต์)
    *   **การกำหนด Label:** หากไฟล์เสียงถูกจัดอยู่ในโฟลเดอร์ย่อย จะใช้ชื่อโฟลเดอร์ย่อยนั้นเป็น `label` ในไฟล์ CSV (ถ้าไม่มีโฟลเดอร์ย่อย หรือไฟล์อยู่ในระดับบนสุดของ `input_dir` Label จะเป็น "unknown") หากไม่มี Label ที่มีความหมาย (เช่น มีแต่ "unknown") คอลัมน์ `label` จะไม่ถูกสร้างขึ้น
*   **การรัน (ตัวอย่าง):**
    ```bash
    # สแกนโฟลเดอร์ placeholder_audio/ (ค่าเริ่มต้น) และสร้าง custom_audio_dataset.csv
    python DatasetAudio/create_audio_dataset_csv.py

    # สแกนโฟลเดอร์อื่นแบบไม่ recursive และตั้งชื่อไฟล์ผลลัพธ์
    python DatasetAudio/create_audio_dataset_csv.py --input_dir path/to/my/audio --output_filename my_audio_files.csv --no-recursive
    ```

### การสร้างข้อมูลเสียงด้วย Hugging Face API (สคริปต์ `gen_*.py`)

สคริปต์เหล่านี้ (`gen_text_to_speech.py`, `gen_automatic_speech_recognition.py`, etc.) ใช้สำหรับสร้างข้อมูลเสียงประเภทต่างๆ โดยเรียกใช้งาน Hugging Face Inference API (ต้องตั้งค่า `HF_TOKEN` ใน `.env`) หรืออาจต้องติดตั้งไลบรารีเพิ่มเติมสำหรับบาง Task

**การกำหนดค่า:**
*   แก้ไขไฟล์ `DatasetAudio/config_audio.py` เพื่อ:
    *   กำหนด Hugging Face Model ID สำหรับแต่ละ Task (เช่น `TTS_MODEL_ID`, `ASR_MODEL_ID`)
    *   กำหนดจำนวนตัวอย่างที่จะสร้าง (`NUM_SAMPLES_PER_TASK`)
    *   เตรียมข้อมูล Input (เช่น ข้อความใน `TTS_INPUT_TEXTS`, Path ของไฟล์เสียงใน `ASR_INPUT_AUDIO`)
        *   **สำคัญ:** สำหรับ Task ที่ต้องการไฟล์เสียง Input (ASR, Audio Classification, VAD, Audio-to-Audio) ให้สร้างโฟลเดอร์ `placeholder_audio/` ในรากโปรเจกต์ และนำไฟล์เสียง (เช่น `.wav`, `.mp3`) ไปวางไว้ หรือแก้ไข Path ใน `config_audio.py` ให้ถูกต้อง สคริปต์จะพยายามสร้างไฟล์ dummy `.wav` หากตรวจไม่พบและติดตั้ง `pydub` ไว้ (`pip install pydub`)
    *   กำหนดชื่อไฟล์ CSV ผลลัพธ์ (เช่น `TTS_FILENAME`)

**การรันสคริปต์ (ตัวอย่าง):**

```bash
# สร้างชุดข้อมูล Text-to-Speech
python DatasetAudio/gen_text_to_speech.py

# สร้างชุดข้อมูล Automatic Speech Recognition (ต้องมีไฟล์เสียงใน placeholder_audio/)
python DatasetAudio/gen_automatic_speech_recognition.py

# สร้างชุดข้อมูล Audio Classification (ต้องมีไฟล์เสียงใน placeholder_audio/)
python DatasetAudio/gen_audio_classification.py

# สร้างชุดข้อมูล Text-to-Audio (ทดลอง)
python DatasetAudio/gen_text_to_audio.py

# สร้างชุดข้อมูล Voice Activity Detection (ทดลอง/Placeholder)
python DatasetAudio/gen_voice_activity_detection.py

# สร้างชุดข้อมูล Audio-to-Audio (ทดลอง/Placeholder)
python DatasetAudio/gen_audio_to_audio.py
```

**ผลลัพธ์ (สำหรับ `gen_*.py`):**

*   ไฟล์ CSV ที่มีข้อมูล Input และ Output (เช่น Path ไฟล์เสียงที่สร้าง, ข้อความที่ถอดความ, ผลการจำแนก) จะถูกบันทึกใน `DataOutput/`
*   ไฟล์เสียงที่สร้างขึ้น (จาก TTS, Text-to-Audio, Audio-to-Audio) จะถูกบันทึกใน `DataOutput/generated_media/audio/`

**หมายเหตุ (สำหรับ `gen_*.py`):**

*   ความสำเร็จของการเรียก API ขึ้นอยู่กับ Model ที่เลือก, สถานะของ Hugging Face Inference API, และ Token ของคุณ
*   บาง Task (เช่น VAD, Audio-to-Audio) อาจทำงานได้ไม่สมบูรณ์ผ่าน API ทั่วไป และอาจต้องใช้ Library เฉพาะทาง (เช่น `pyannote.audio`, `librosa`, `torchaudio`) และรันโมเดลในเครื่อง

---

## การใช้งานสคริปต์สาธิต (`Script/Dataset/`)

สคริปต์ในโฟลเดอร์ `Script/Dataset/` มีไว้เพื่อสาธิตการใช้งานโมเดลพื้นฐานสำหรับงาน NLP ต่างๆ โดยใช้ข้อมูลตัวอย่างที่กำหนดไว้ในโค้ด (เช่น `Dataset/ner_dataset.py`, `Dataset/emotion_detection_dataset.py`)

**การทำงานทั่วไป:**

1. **โหลดโมเดล:** โหลด Pre-trained Model และ Tokenizer (ส่วนใหญ่ใช้ `airesearch/wangchanberta-base-att-spm-uncased` หรือโมเดลที่เหมาะสมกับงานนั้นๆ)
2. **ประเมินผล (Evaluate):** ประมวลผลชุดข้อมูลตัวอย่าง (`evaluate_model()` หรือฟังก์ชันคล้ายกัน) และแสดงผลลัพธ์การทำนายเทียบกับค่าที่คาดหวัง รวมถึงค่า Accuracy (ถ้ามี)
3. **สาธิต (Demonstrate):** แสดงตัวอย่างการทำงานของโมเดลกับข้อมูลบางส่วน (`demonstrate_usage()`)
4. **โหมดโต้ตอบ (Interactive):** ส่วนใหญ่มีโหมดให้ผู้ใช้ป้อนข้อความและดูผลการทำนาย (`run_interactive_demo()`)
5. **วิเคราะห์รูปแบบ (Analyze Patterns):** บางสคริปต์อาจมีการวิเคราะห์รูปแบบเฉพาะของงานนั้นๆ (`analyze_..._patterns()`)

**สคริปต์ที่น่าสนใจ:**

### Content Moderation (`Script/Dataset/content_moderation.py`)

* **หน้าที่:** ตรวจสอบเนื้อหาที่ไม่เหมาะสม (เช่น คำหยาบ, ความรุนแรง)
* **การรัน:** `python Script/Dataset/content_moderation.py`
* **ผลลัพธ์:** แสดงผลการทำนาย, ความมั่นใจ, และเปรียบเทียบกับ Label ที่ถูกต้อง มีโหมด Interactive

### Conversation Simulation (`Script/Dataset/conversation_simulation.py`)

* **หน้าที่:** จำลองบทสนทนา, วิเคราะห์ประเภท, อารมณ์, และสร้างการตอบกลับ
* **การรัน:** `python Script/Dataset/conversation_simulation.py`
* **ผลลัพธ์:** แสดงการวิเคราะห์บทสนทนาตัวอย่าง, การสร้าง Response, และมีโหมด Interactive

### Emotion Detection (`Script/Dataset/emotion_detection.py`)

* **หน้าที่:** วิเคราะห์อารมณ์ในข้อความภาษาไทย
* **การรัน:** `python Script/Dataset/emotion_detection.py`
* **ผลลัพธ์:** แสดงผลการทำนายอารมณ์, ความมั่นใจ, และเปรียบเทียบกับ Label ที่ถูกต้อง มีโหมด Interactive

### FAQ Summarization (`Script/Dataset/faq_summarization.py`)

* **หน้าที่:** สร้างคำตอบสำหรับคำถามจากเอกสาร FAQ และสรุปประเด็นสำคัญ
* **การรัน:** `python Script/Dataset/faq_summarization.py`
* **ผลลัพธ์:** แสดงคำตอบที่สร้างขึ้นเทียบกับคำตอบตัวอย่าง และแสดง Key Points ที่สกัดได้ มีโหมด Interactive

### NER Handler (`Script/Dataset/ner_handler.py`)

* **หน้าที่:** ระบุ Entities (เช่น บุคคล, องค์กร, สถานที่) ในข้อความ
* **การรัน:** `python Script/Dataset/ner_handler.py`
* **ผลลัพธ์:** แสดง Entities ที่ตรวจพบพร้อมประเภทและความมั่นใจ คำนวณ Accuracy และมีโหมด Interactive

### Paraphrase Identification (`Script/Dataset/paraphrase_identification.py`)

* **หน้าที่:** ตรวจสอบว่าประโยคสองประโยคมีความหมายเหมือนกันหรือไม่ และสร้างประโยคใหม่ที่มีความหมายคล้ายกัน
* **การรัน:** `python Script/Dataset/paraphrase_identification.py`
* **ผลลัพธ์:** แสดงผลการเปรียบเทียบ, คะแนนความคล้ายคลึง, การวิเคราะห์ความแตกต่าง, และตัวอย่างประโยคที่สร้างขึ้นใหม่ มีโหมด Interactive

### Style Transfer (`Script/Dataset/style_transfer.py`)

* **หน้าที่:** แปลงรูปแบบภาษาระหว่างทางการ (Formal) และไม่ทางการ (Informal)
* **การรัน:** `python Script/Dataset/style_transfer.py`
* **ผลลัพธ์:** แสดงผลการแปลงทั้งสองทิศทาง (Formal -> Informal, Informal -> Formal) และการวิเคราะห์รูปแบบ มีโหมด Interactive

### Transcript Handler (`Script/Dataset/transcript_handler.py`)

* **หน้าที่:** วิเคราะห์บทสนทนาหรือข้อความที่ถอดความจากเสียง ระบุสภาพแวดล้อม, ผู้พูด, รูปแบบการพูด, และทำความสะอาดข้อความ
* **การรัน:** `python Script/Dataset/transcript_handler.py`
* **ผลลัพธ์:** แสดงผลการวิเคราะห์ต่างๆ เช่น สภาพแวดล้อม, ผู้พูด, รูปแบบคำพูด, ความเป็นทางการ มีโหมด Interactive

### TTS Script Handler (`Script/Dataset/tts_script_handler.py`)

* **หน้าที่:** วิเคราะห์สคริปต์สำหรับ Text-to-Speech (TTS) เพื่อระบุประเภทการพูด, อารมณ์, และแนะนำพารามิเตอร์
* **การรัน:** `python Script/Dataset/tts_script_handler.py`
* **ผลลัพธ์:** แสดงผลการวิเคราะห์อารมณ์, คุณสมบัติข้อความ, และคำแนะนำสำหรับ TTS มีโหมด Interactive

---

## ผลลัพธ์ (Outputs)

* **โมเดลที่ดาวน์โหลด:** จะถูกเก็บไว้ในโฟลเดอร์ `Model/`
* **ชุดข้อมูลที่สร้าง (NLP, Vision, Audio):** ไฟล์ CSV จะถูกบันทึกในโฟลเดอร์ `DataOutput/` (สร้างขึ้นอัตโนมัติหากยังไม่มี)
* **ไฟล์ Media ที่สร้าง (Vision, Audio):** ไฟล์รูปภาพ, วิดีโอ, หรือเสียง ที่สร้างโดยสคริปต์ Vision/Audio จะถูกบันทึกใน `DataOutput/generated_media/` โดยแยกตามโฟลเดอร์ย่อยของแต่ละ Task (เช่น `audio/`, `text_to_image/`)

---

หากพบปัญหาหรือมีคำถามเพิ่มเติม โปรดเปิด Issue ใน GitHub repository ของโปรเจกต์
