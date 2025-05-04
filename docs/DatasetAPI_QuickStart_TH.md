# คู่มือเริ่มต้นใช้งาน DatasetAPI

## การติดตั้ง

1. Clone repository และติดตั้ง dependencies:
```bash
git clone <repository-url>
cd aibuilderPart1
pip install -r requirements.txt
```

2. ตั้งค่า API key:
- สร้างไฟล์ .env
- เพิ่ม DEEPSEEK_API_KEY=your_api_key_here

## การใช้งาน

### รันระบบ

มี 2 วิธีในการใช้งาน:

1. ผ่าน Command Line:
```bash
python DatasetAPI/run_apigen_simulation.py
```

2. ผ่าน Streamlit Web Interface:
```bash
streamlit run DatasetAPI/streamlit_app.py
```

การใช้งานผ่าน Streamlit จะมีหน้า web interface ให้:
- ตั้งค่าพารามิเตอร์ต่างๆ ได้ง่าย
- ดูรายการ API ที่รองรับ
- ดูผลลัพธ์แบบ real-time
- ดูกราฟและสถิติ

### ผลลัพธ์
ระบบจะสร้างไฟล์ต่างๆ ใน DataOutput/apigen_llm_diverse/:
- ข้อมูลหลัก: verified_api_calls_llm_diverse.jsonl
- ข้อมูล CSV: verified_api_calls_llm_diverse.csv
- ข้อมูลอย่างง่าย: simplified_api_calls_llm_diverse.jsonl
- สถิติ: api_call_statistics.json
- กราฟ: visualizations/
- Log: logs/apigen_llm.log

### การปรับแต่ง
แก้ไขการตั้งค่าใน config_apigen_llm.py:
```python
# จำนวนข้อมูลที่ต้องการสร้าง
NUM_GENERATIONS_TO_ATTEMPT = 50

# ความหลากหลายของข้อมูล (0.0-1.0)
GENERATION_TEMPERATURE = 0.8

# เปิด/ปิดฟีเจอร์
ENABLE_FORMAT_CHECK = True     # ตรวจสอบรูปแบบข้อมูล
ENABLE_EXECUTION_CHECK = True  # จำลองการเรียก API
ENABLE_SEMANTIC_CHECK = True   # ตรวจสอบความหมาย
USE_THAI_QUERIES = True       # สร้างคำถามภาษาไทย
```

## API ที่รองรับ

### การเงิน
- get_stock_price: ดูราคาหุ้น
- transfer_money: โอนเงิน
- get_account_balance: เช็คยอดเงิน
- get_transaction_history: ดูประวัติธุรกรรม

### สุขภาพ
- find_nearby_doctors: ค้นหาหมอใกล้ตัว
- book_medical_appointment: นัดหมอ

### เครื่องมือ
- set_timer: ตั้งนาฬิกาจับเวลา
- calculate: คำนวณ
- set_reminder: ตั้งการแจ้งเตือน

### ภาษาไทย
- translate_th_en: แปลไทย-อังกฤษ
- find_thai_food: ค้นหาอาหารไทย

## การตรวจสอบคุณภาพ

ระบบมีการตรวจสอบ 3 ขั้นตอน:
1. ตรวจสอบรูปแบบข้อมูล
   - โครงสร้าง JSON
   - ประเภทข้อมูล
   - พารามิเตอร์ที่จำเป็น

2. จำลองการเรียก API
   - ทดสอบการทำงาน
   - จัดการข้อผิดพลาด

3. ตรวจสอบความหมาย
   - ตรวจสอบคำสำคัญ
   - ความสอดคล้องของคำถามและ API

## การแก้ไขปัญหา

1. หากพบ error "DEEPSEEK_API_KEY not set":
   - ตรวจสอบไฟล์ .env
   - ตรวจสอบค่า API key

2. หากไม่มีผลลัพธ์:
   - ตรวจสอบ logs ใน DataOutput/apigen_llm_diverse/logs/
   - ลองลดจำนวน NUM_GENERATIONS_TO_ATTEMPT

3. หากต้องการดูรายละเอียดเพิ่มเติม:
   - ดูเอกสารฉบับเต็มที่ docs/DatasetAPI.md
   - ตรวจสอบ comments ในโค้ด