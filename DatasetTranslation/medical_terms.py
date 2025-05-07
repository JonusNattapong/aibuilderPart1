import re

# Medical terms mapping (English to Thai)
MEDICAL_TERMS = {
    "myocardial infarction": "กล้ามเนื้อหัวใจขาดเลือด",
    "chest pain": "อาการเจ็บหน้าอก",
    "MRI": "เอ็มอาร์ไอ",
    "multiple sclerosis": "โรคปลอกประสาทเสื่อมแข็ง",
    "cerebral cortex": "เปลือกสมอง",
    "Parkinson's disease": "โรคพาร์กินสัน",
    "tremors": "อาการสั่น",
    "bradykinesia": "ภาวะการเคลื่อนไหวช้า",
    "acute": "เฉียบพลัน",
    "chronic": "เรื้อรัง",
    "lesions": "รอยโรค",
    "symptoms": "อาการ",
    "diagnosis": "การวินิจฉัย",
    "prognosis": "การพยากรณ์โรค"
}

# Anatomical terms
ANATOMICAL_TERMS = {
    "heart": "หัวใจ",
    "brain": "สมอง",
    "lung": "ปอด",
    "liver": "ตับ",
    "kidney": "ไต",
    "spine": "กระดูกสันหลัง",
    "muscle": "กล้ามเนื้อ",
    "bone": "กระดูก",
    "joint": "ข้อต่อ",
    "artery": "หลอดเลือดแดง",
    "vein": "หลอดเลือดดำ"
}

# Symptom terms
SYMPTOM_TERMS = {
    "pain": "ความเจ็บปวด",
    "fever": "ไข้",
    "cough": "ไอ",
    "nausea": "คลื่นไส้",
    "vomiting": "อาเจียน",
    "dizziness": "วิงเวียน",
    "fatigue": "อ่อนเพลีย",
    "headache": "ปวดศีรษะ",
    "inflammation": "การอักเสบ",
    "swelling": "การบวม"
}

# Compound anatomical terms
COMPOUND_ANATOMICAL = {
    "cardiovascular": "หัวใจและหลอดเลือด",
    "musculoskeletal": "กล้ามเนื้อและกระดูก",
    "gastrointestinal": "ทางเดินอาหาร",
    "neurological": "ระบบประสาท",
    "respiratory": "ระบบหายใจ"
}

def escape_regex_chars(text: str) -> str:
    """Escape special regex characters in text."""
    return re.escape(text)

def replace_all_terms(text: str) -> str:
    """Replace all medical terms in text with their Thai equivalents."""
    
    # Combine all term dictionaries
    all_terms = {
        **COMPOUND_ANATOMICAL,  # Check compound terms first
        **MEDICAL_TERMS,
        **ANATOMICAL_TERMS,
        **SYMPTOM_TERMS
    }
    
    # Sort terms by length (longest first) to handle overlapping terms
    sorted_terms = sorted(all_terms.keys(), key=len, reverse=True)
    
    # Replace each term
    result = text
    for term in sorted_terms:
        pattern = re.compile(escape_regex_chars(term), re.IGNORECASE)
        if pattern.search(text):
            result = pattern.sub(all_terms[term], result)
            
    return result