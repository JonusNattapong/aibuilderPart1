"""
Simulator for API execution logic.
These functions mimic calling real APIs.
"""
import random
import re
from datetime import datetime, timedelta

# --- Finance APIs ---
def execute_get_stock_price(ticker_symbol):
    """Simulates getting stock price."""
    if not isinstance(ticker_symbol, str) or not re.match(r'^[A-Z]{1,5}$', ticker_symbol):
        return False, {"error": "Invalid ticker symbol"}
    
    if random.random() < 0.05:  # 5% chance of failure
        return False, {"error": "Market data temporarily unavailable"}
    
    price = round(random.uniform(10, 1000), 2)
    return True, {"ticker": ticker_symbol, "price": price, "currency": "USD"}

def execute_transfer_money(from_account_id, to_account_id, amount, currency="THB"):
    """Simulates money transfer."""
    if not all(isinstance(acc, str) for acc in [from_account_id, to_account_id]):
        return False, {"error": "Invalid account ID format"}
    if not isinstance(amount, (int, float)) or amount <= 0:
        return False, {"error": "Invalid transfer amount"}
    if currency not in ["USD", "THB", "EUR", "JPY"]:
        return False, {"error": "Unsupported currency"}

    if random.random() < 0.08:  # 8% chance of failure
        return False, {"error": "Insufficient funds or network error"}
    
    return True, {
        "transaction_id": f"TR{random.randint(10000, 99999)}",
        "status": "completed",
        "amount": amount,
        "currency": currency
    }

def execute_get_account_balance(account_id, include_pending=False):
    """Simulates getting account balance."""
    if not isinstance(account_id, str):
        return False, {"error": "Invalid account ID"}
    
    if random.random() < 0.03:  # 3% chance of failure
        return False, {"error": "Account lookup failed"}
    
    balance = round(random.uniform(1000, 100000), 2)
    pending = round(random.uniform(-1000, 1000), 2) if include_pending else 0
    return True, {
        "account_id": account_id,
        "balance": balance,
        "pending": pending if include_pending else None,
        "currency": "THB"
    }

def execute_get_transaction_history(account_id, start_date=None, end_date=None, max_transactions=20):
    """Simulates getting transaction history."""
    if not isinstance(account_id, str):
        return False, {"error": "Invalid account ID"}
    try:
        max_transactions = min(int(max_transactions), 100)
    except (ValueError, TypeError):
        max_transactions = 20

    if random.random() < 0.05:  # 5% chance of failure
        return False, {"error": "Transaction lookup failed"}

    transactions = []
    for _ in range(random.randint(1, max_transactions)):
        amount = round(random.uniform(-5000, 5000), 2)
        transactions.append({
            "id": f"TX{random.randint(10000, 99999)}",
            "date": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
            "amount": amount,
            "type": "debit" if amount < 0 else "credit"
        })
    return True, {"account_id": account_id, "transactions": transactions}

# --- Health APIs ---
def execute_find_nearby_doctors(location, specialty=None, radius_km=5):
    """Simulates finding nearby doctors."""
    if not isinstance(location, str) or not location:
        return False, {"error": "Invalid location"}
    if radius_km and (not isinstance(radius_km, (int, float)) or not 1 <= radius_km <= 50):
        return False, {"error": "Invalid radius"}

    if random.random() < 0.05:  # 5% chance of failure
        return False, {"error": "Location service unavailable"}

    doctors = []
    num_results = random.randint(1, 5)
    specialties = ["Dentist", "Cardiologist", "Pediatrician", "General Practice"] if not specialty else [specialty]
    
    for i in range(num_results):
        doctors.append({
            "id": f"DOC{random.randint(1000, 9999)}",
            "name": f"Dr. Smith {i+1}",
            "specialty": random.choice(specialties),
            "distance": round(random.uniform(0.1, radius_km), 1)
        })
    return True, {"doctors": doctors, "location": location}

def execute_book_medical_appointment(doctor_id, appointment_date, appointment_time, patient_name, reason=None):
    """Simulates booking a medical appointment."""
    if not all(isinstance(x, str) for x in [doctor_id, appointment_date, appointment_time, patient_name]):
        return False, {"error": "Invalid parameter types"}
    
    try:
        datetime.strptime(f"{appointment_date} {appointment_time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return False, {"error": "Invalid date/time format"}

    if random.random() < 0.15:  # 15% chance of failure (time slot might be taken)
        return False, {"error": "Selected time slot is not available"}
    
    return True, {
        "appointment_id": f"APT{random.randint(10000, 99999)}",
        "status": "confirmed",
        "doctor_id": doctor_id,
        "patient_name": patient_name,
        "datetime": f"{appointment_date} {appointment_time}"
    }

# --- Tools APIs ---
def execute_set_timer(duration, label=None):
    """Simulates setting a timer."""
    if not isinstance(duration, str) or not duration:
        return False, {"error": "Invalid duration format"}

    if random.random() < 0.02:  # 2% chance of failure
        return False, {"error": "Timer service unavailable"}

    timer_id = f"TMR{random.randint(1000, 9999)}"
    return True, {
        "timer_id": timer_id,
        "duration": duration,
        "label": label,
        "status": "started"
    }

def execute_calculate(expression):
    """Simulates mathematical calculation."""
    if not isinstance(expression, str) or not expression:
        return False, {"error": "Invalid expression"}
    
    # Very basic simulation - in reality would need proper parsing
    try:
        # IMPORTANT: This is just for simulation. Never eval() user input in production!
        result = random.uniform(0, 100)  # Simulate a result instead of actually calculating
        return True, {"expression": expression, "result": round(result, 2)}
    except Exception as e:
        return False, {"error": "Invalid mathematical expression"}

def execute_set_reminder(message, time, priority="medium"):
    """Simulates setting a reminder."""
    if not isinstance(message, str) or not message:
        return False, {"error": "Invalid message"}
    if not isinstance(time, str) or not time:
        return False, {"error": "Invalid time"}
    if priority not in ["low", "medium", "high"]:
        priority = "medium"

    if random.random() < 0.03:  # 3% chance of failure
        return False, {"error": "Reminder service temporarily unavailable"}

    return True, {
        "reminder_id": f"RMD{random.randint(1000, 9999)}",
        "message": message,
        "time": time,
        "priority": priority,
        "status": "set"
    }

# --- Thai-specific APIs ---
def execute_translate_th_en(text, formal=False):
    """Simulates Thai to English translation."""
    if not isinstance(text, str) or not text:
        return False, {"error": "Invalid text"}

    if random.random() < 0.05:  # 5% chance of failure
        return False, {"error": "Translation service unavailable"}

    # Simulate translation by returning a placeholder
    translated = f"[EN: {text[:30]}{'...' if len(text) > 30 else ''}]"
    return True, {
        "original": text,
        "translated": translated,
        "formal": formal
    }

def execute_find_thai_food(location, dish_type=None, spice_level=None, max_price=None):
    """Simulates finding Thai food locations."""
    if not isinstance(location, str) or not location:
        return False, {"error": "Invalid location"}
    if spice_level and spice_level not in ["mild", "medium", "spicy", "very_spicy"]:
        return False, {"error": "Invalid spice level"}
    if max_price and (not isinstance(max_price, (int, float)) or max_price < 0):
        return False, {"error": "Invalid price"}

    if random.random() < 0.05:  # 5% chance of failure
        return False, {"error": "Restaurant search unavailable"}

    results = []
    num_results = random.randint(1, 5)
    dishes = ["Pad Thai", "Tom Yum", "Green Curry", "Som Tum"] if not dish_type else [dish_type]
    
    for i in range(num_results):
        price = random.randint(50, 500)
        if max_price and price > max_price:
            continue
        results.append({
            "name": f"Thai Restaurant {i+1}",
            "dish": random.choice(dishes),
            "price": price,
            "spice_level": spice_level or random.choice(["mild", "medium", "spicy", "very_spicy"]),
            "distance": f"{random.randint(1, 20)} km"
        })
    
    return True, {"location": location, "restaurants": results}

# Map API names to executor functions
API_EXECUTORS = {
    # Finance
    "get_stock_price": execute_get_stock_price,
    "transfer_money": execute_transfer_money,
    "get_account_balance": execute_get_account_balance,
    "get_transaction_history": execute_get_transaction_history,
    
    # Health
    "find_nearby_doctors": execute_find_nearby_doctors,
    "book_medical_appointment": execute_book_medical_appointment,
    
    # Tools
    "set_timer": execute_set_timer,
    "calculate": execute_calculate,
    "set_reminder": execute_set_reminder,
    
    # Thai-specific
    "translate_th_en": execute_translate_th_en,
    "find_thai_food": execute_find_thai_food
}
