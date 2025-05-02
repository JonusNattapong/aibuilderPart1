import os

# --- General Configuration ---
MODEL_ID = "scb10x/llama3.2-typhoon2-3b-instruct"
OUTPUT_DIR = 'DataOutput' # Relative to the project root or where the main script is run from
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds
NUM_SAMPLES_PER_TASK = 15 # Default number of samples to generate for each task

# --- DeepSeek API Configuration ---
# It's strongly recommended to set the API key via environment variable:
# export DEEPSEEK_API_KEY='your_api_key'
# Or in Python before running the script: os.environ["DEEPSEEK_API_KEY"] = "your_api_key"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat" # Or "deepseek-coder" or other available models

# --- Task-Specific Configuration ---

# Text Classification
CLASSIFICATION_TOPICS = ["กีฬา", "เทคโนโลยี", "สุขภาพ", "การเดินทาง", "อาหาร", "การเมือง", "บันเทิง", "ธุรกิจ", "การศึกษา"]
CLASSIFICATION_CATEGORIES = ["กีฬา", "เทคโนโลยี", "สุขภาพ", "การเดินทาง", "อาหาร", "การเมือง", "บันเทิง", "ธุรกิจ", "การศึกษา", "สังคม", "สิ่งแวดล้อม"]
CLASSIFICATION_FILENAME = "generated_classification_data.csv"

# Question Answering (Extractive)
QA_TOPICS = ["วิทยาศาสตร์", "ประวัติศาสตร์ไทย", "ภูมิศาสตร์โลก", "เทคโนโลยีอวกาศ", "สิ่งแวดล้อม", "วรรณกรรมไทย", "บุคคลสำคัญ"]
QA_FILENAME = "generated_qa_data.csv"

# Table Question Answering
TABLE_QA_TOPICS = ["ข้อมูลประชากร", "สถิติกีฬา", "ราคาสินค้าเกษตร", "ตารางเที่ยวบิน", "ผลประกอบการบริษัท"]
TABLE_QA_FILENAME = "generated_table_qa_data.csv"

# Zero-Shot Classification Data
ZERO_SHOT_TOPICS = ["ข่าวรอบวัน", "รีวิวสินค้า", "กระทู้ถามตอบ", "เนื้อหาโซเชียลมีเดีย"]
ZERO_SHOT_POTENTIAL_LABELS = ["เศรษฐกิจ", "การเมือง", "สังคม", "กีฬา", "บันเทิง", "เทคโนโลยี", "สุขภาพ", "อาหาร", "ท่องเที่ยว", "การศึกษา"] # Example broad labels
ZERO_SHOT_FILENAME = "generated_zero_shot_data.csv"

# Token Classification (NER)
NER_TOPICS = ["ข่าวธุรกิจ", "เหตุการณ์ปัจจุบัน", "รีวิวภาพยนตร์", "บทความท่องเที่ยว", "รายงานการประชุม", "ประวัติบุคคล"]
NER_FILENAME = "generated_ner_data.csv"

# Translation (Thai to English)
TRANSLATION_TOPICS = ["การสนทนาทั่วไป", "สำนวนไทย", "คำศัพท์เทคนิค", "เมนูอาหาร", "ป้ายประกาศ", "เนื้อเพลง"]
TRANSLATION_FILENAME = "generated_translation_th_en.csv"

# Summarization
SUMMARIZATION_TOPICS = ["บทความวิชาการ", "ข่าวต่างประเทศ", "รีวิวหนังสือ", "ขั้นตอนการใช้งาน", "พอดแคสต์", "คำปราศรัย"]
SUMMARIZATION_FILENAME = "generated_summarization_data.csv"

# Feature Extraction / Sentence Similarity
SENTENCE_SIMILARITY_TOPICS = ["หัวข้อข่าวเดียวกัน", "ความหมายใกล้เคียงกัน", "ความหมายตรงข้ามกัน", "ประโยคที่ไม่เกี่ยวข้องกัน"]
SENTENCE_SIMILARITY_FILENAME = "generated_sentence_similarity_data.csv"

# Text Generation
TEXT_GEN_TOPICS = ["การเขียนอีเมล", "การแต่งกลอน", "การสร้างสโลแกน", "การเขียนบทสนทนา", "การเขียนรีวิวสินค้า", "การเล่าเรื่อง"]
TEXT_GEN_FILENAME = "generated_text_generation_data.csv"

# Style Transfer (Formal/Informal)
STYLE_TRANSFER_TOPICS = ["การแจ้งข่าว", "การขอความช่วยเหลือ", "การแสดงความยินดี", "การปฏิเสธ", "การเชิญชวน", "การแสดงความคิดเห็น"]
STYLE_TRANSFER_FILENAME = "generated_style_transfer_data.csv"

# Fill-Mask
FILL_MASK_TOPICS = ["สุภาษิตไทย", "ความรู้ทั่วไป", "เนื้อเพลง", "ประโยคบอกเล่า"]
FILL_MASK_FILENAME = "generated_fill_mask_data.csv"

# Text Ranking
TEXT_RANKING_TOPICS = ["สถานที่ท่องเที่ยว", "สูตรอาหาร", "ข้อมูลทางประวัติศาสตร์", "วิธีการทำ DIY"]
TEXT_RANKING_FILENAME = "generated_text_ranking_data.csv"

# --- Helper to ensure OUTPUT_DIR exists ---
def ensure_output_dir():
    """Creates the output directory if it doesn't exist."""
    # Consider making OUTPUT_DIR an absolute path or relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', '..', OUTPUT_DIR) # Assumes OUTPUT_DIR is relative to project root
    os.makedirs(output_path, exist_ok=True)
    return output_path # Return the calculated absolute path

# You might want to adjust the path calculation based on your project structure
# If OUTPUT_DIR should be directly inside 'Generate', use:
# output_path = os.path.join(script_dir, OUTPUT_DIR)

# If OUTPUT_DIR is relative to where you RUN the main script,
# the os.makedirs in the main script is sufficient.
# This function provides an alternative if you want paths relative to the config file.
