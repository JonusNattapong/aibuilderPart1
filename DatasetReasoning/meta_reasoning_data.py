# -*- coding: utf-8 -*-
import uuid

meta_reasoning_data = [
    # --- 7. Meta-Reasoning ---
    {
        "id": str(uuid.uuid4()),
        "task": "Strategy Selection",
        "reasoning_type_tested": "Meta-Reasoning",
        "input_data": {
            "problem_description": "ต้องการหาข้อมูลล่าสุดเกี่ยวกับอัตราแลกเปลี่ยนเงินบาทเทียบกับดอลลาร์สหรัฐ",
            "available_strategies": ["Chain of Thought", "ReAct (with Search Tool)", "Program-aided Reasoning"]
        },
        "expected_output": {
            "reasoning": "ข้อมูลอัตราแลกเปลี่ยนเป็นข้อมูลแบบเรียลไทม์และต้องใช้แหล่งข้อมูลภายนอก CoT และ PaR ไม่เหมาะ ReAct ที่มีเครื่องมือค้นหา (Search Tool) เหมาะสมที่สุด",
            "selected_strategy": "ReAct (with Search Tool)"
        }
    },
]
