"""
Configuration file for APIGen using LLM.
Contains definitions of available APIs covering diverse categories.
"""
import os
import logging
import json # Added for loading external APIs
from typing import Dict, List, Any

# --- API Library Configuration ---

# Option 1: Define API Library directly in this file (current method)
API_LIBRARY_INTERNAL = {
    # Finance (11.7%)
    "get_stock_price": {
        "description": "Get the current stock price for a given ticker symbol.",
        "parameters": {
            "ticker_symbol": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL, GOOG)", "required": True},
        },
        "category": "finance",
        "plausibility_checks": { # Example plausibility check rules
            "ticker_symbol": {"regex": r"^[A-Z]{1,5}$"} # Simple check for 1-5 uppercase letters
        }
    },
    "transfer_money": {
        "description": "Transfer money between accounts.",
        "parameters": {
            "from_account_id": {"type": "string", "description": "The source account ID", "required": True},
            "to_account_id": {"type": "string", "description": "The destination account ID", "required": True},
            "amount": {"type": "number", "description": "The amount to transfer", "required": True},
            "currency": {"type": "string", "description": "Currency code (e.g., USD, THB)", "required": False, "default": "THB"}
        },
        "category": "finance",
        "plausibility_checks": {
            "amount": {"min_value": 0.01}, # Amount must be positive
            "currency": {"allowed_values": ["USD", "THB", "EUR", "JPY"]} # Example allowed currencies
        }
    },
    "get_account_balance": {
        "description": "Get the current balance of a financial account.",
        "parameters": {
            "account_id": {"type": "string", "description": "The account ID", "required": True},
            "include_pending": {"type": "boolean", "description": "Whether to include pending transactions", "required": False, "default": False}
        },
        "category": "finance"
        # No specific plausibility checks added here for brevity
    },
    "get_transaction_history": {
        "description": "Get transaction history for a financial account.",
        "parameters": {
            "account_id": {"type": "string", "description": "The account ID", "required": True},
            "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format", "required": False},
            "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format", "required": False},
            "max_transactions": {"type": "integer", "description": "Maximum number of transactions to return", "required": False, "default": 20}
        },
        "category": "finance",
        "plausibility_checks": {
             "start_date": {"regex": r"^\d{4}-\d{2}-\d{2}$"}, # Check YYYY-MM-DD format
             "end_date": {"regex": r"^\d{4}-\d{2}-\d{2}$"},   # Check YYYY-MM-DD format
             "max_transactions": {"min_value": 1, "max_value": 100} # Limit max transactions
        }
    },
    # Health (expanded)
    "find_nearby_doctors": {
        "description": "Find doctors near a specific location.",
        "parameters": {
            "location": {"type": "string", "description": "City or address to search near", "required": True},
            "specialty": {"type": "string", "description": "Medical specialty (e.g., Dentist, Cardiologist)", "required": False},
            "radius_km": {"type": "number", "description": "Search radius in kilometers", "required": False, "default": 5}
        },
        "category": "health",
        "plausibility_checks": {
            "radius_km": {"min_value": 1, "max_value": 50}
        }
    },
    "book_medical_appointment": {
        "description": "Book a medical appointment with a healthcare provider.",
        "parameters": {
            "doctor_id": {"type": "string", "description": "ID of the healthcare provider", "required": True},
            "appointment_date": {"type": "string", "description": "Preferred date (YYYY-MM-DD)", "required": True},
            "appointment_time": {"type": "string", "description": "Preferred time (HH:MM)", "required": True},
            "patient_name": {"type": "string", "description": "Name of the patient", "required": True},
            "reason": {"type": "string", "description": "Reason for the appointment", "required": False}
        },
        "category": "health",
         "plausibility_checks": {
             "appointment_date": {"regex": r"^\d{4}-\d{2}-\d{2}$"},
             "appointment_time": {"regex": r"^\d{2}:\d{2}$"} # Basic HH:MM check
         }
    },
    # Tools (expanded)
    "set_timer": {
        "description": "Set a timer for a specified duration.",
        "parameters": {
            "duration": {"type": "string", "description": "Duration (e.g., '5 minutes', '1 hour 30 seconds')", "required": True},
            "label": {"type": "string", "description": "Optional label for the timer", "required": False}
        },
        "category": "tools"
        # Plausibility check for duration string could be complex, skipped for now
    },
    "calculate": {
        "description": "Perform a mathematical calculation.",
        "parameters": {
            "expression": {"type": "string", "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')", "required": True},
        },
        "category": "tools"
        # Plausibility check for expression could involve trying to parse it, skipped
    },
    "set_reminder": {
        "description": "Set a reminder for a specific time or after a delay.",
        "parameters": {
            "message": {"type": "string", "description": "The reminder message", "required": True},
            "time": {"type": "string", "description": "Time for the reminder (e.g., '2023-12-25 08:00', 'tomorrow at 9am', 'in 2 hours')", "required": True},
            "priority": {"type": "string", "description": "Reminder priority (low, medium, high)", "required": False, "default": "medium"}
        },
        "category": "tools",
        "plausibility_checks": {
            "priority": {"allowed_values": ["low", "medium", "high"]}
        }
    },
    # Thai-specific APIs
    "translate_th_en": {
        "description": "Translate text from Thai to English.",
        "parameters": {
            "text": {"type": "string", "description": "The Thai text to translate", "required": True},
            "formal": {"type": "boolean", "description": "Whether to use formal language in translation", "required": False, "default": False}
        },
        "category": "translation"
    },
    "find_thai_food": {
        "description": "Find Thai restaurants or food dishes.",
        "parameters": {
            "location": {"type": "string", "description": "Location to search near", "required": True},
            "dish_type": {"type": "string", "description": "Specific Thai dish or food category", "required": False},
            "spice_level": {"type": "string", "description": "Preferred spice level (mild, medium, spicy, very_spicy)", "required": False},
            "max_price": {"type": "number", "description": "Maximum price (in THB)", "required": False}
        },
        "category": "food",
        "plausibility_checks": {
            "spice_level": {"allowed_values": ["mild", "medium", "spicy", "very_spicy"]},
            "max_price": {"min_value": 0}
        }
    }
}

# Option 2: Load API Library from an external JSON file
# Set API_DEFINITIONS_FILE to the path of your JSON file relative to the project root
# e.g., API_DEFINITIONS_FILE = "DatasetAPI/api_definitions.json"
API_DEFINITIONS_FILE = None # Set to None to use API_LIBRARY_INTERNAL

API_LIBRARY = {}
if API_DEFINITIONS_FILE:
    try:
        definitions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), API_DEFINITIONS_FILE)
        with open(definitions_path, 'r', encoding='utf-8') as f:
            API_LIBRARY = json.load(f)
        print(f"Loaded API definitions from: {definitions_path}")
    except Exception as e:
        print(f"Warning: Could not load API definitions from {API_DEFINITIONS_FILE}. Using internal library. Error: {e}")
        API_LIBRARY = API_LIBRARY_INTERNAL
else:
    API_LIBRARY = API_LIBRARY_INTERNAL


# --- Output Configuration ---
# Determine base path assuming this file is in DatasetAPI
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directory and file naming
SIMULATED_OUTPUT_DIR = os.path.join(BASE_PATH, "DataOutput", "apigen_llm_diverse")
VISUALIZATIONS_DIR = os.path.join(SIMULATED_OUTPUT_DIR, "visualizations")
LOG_DIR = os.path.join(SIMULATED_OUTPUT_DIR, "logs")
CACHE_DIR = os.path.join(SIMULATED_OUTPUT_DIR, "cache")

# Output files
VERIFIED_DATA_FILENAME = "verified_api_calls_llm_diverse.jsonl"
VERIFIED_DATA_CSV_FILENAME = "verified_api_calls_llm_diverse.csv"
SIMPLIFIED_DATA_FILENAME = "simplified_api_calls_llm_diverse.jsonl"
STATISTICS_FILENAME = "api_call_statistics.json"
LOG_FILENAME = "apigen_llm.log"
CACHE_FILENAME = "llm_response_cache.json"
FEW_SHOT_EXAMPLES_FILENAME = "verified_api_calls_llm_diverse.jsonl" # Use existing verified data for few-shot

# --- Semantic Checker Keywords ---
# Map keywords in query to expected API function names for basic semantic check
SEMANTIC_KEYWORDS_TO_API = {
    # Finance
    "stock": "get_stock_price", "price of": "get_stock_price", "shares": "get_stock_price",
    "transfer": "transfer_money", "send money": "transfer_money", "wire": "transfer_money",
    "balance": "get_account_balance", "how much money": "get_account_balance",
    "transaction": "get_transaction_history", "payment history": "get_transaction_history",
    
    # Health
    "doctor": "find_nearby_doctors", "clinic": "find_nearby_doctors", "hospital": "find_nearby_doctors", 
    "dentist": "find_nearby_doctors", "appointment": "book_medical_appointment", 
    "checkup": "book_medical_appointment", "medical visit": "book_medical_appointment",
    
    # Tools
    "timer": "set_timer", "stopwatch": "set_timer", "alarm": "set_timer",
    "calculate": "calculate", "math": "calculate", "equation": "calculate", "solve": "calculate",
    "remind": "set_reminder", "reminder": "set_reminder", "don't forget": "set_reminder",
    
    # Thai-specific
    "translate": "translate_th_en", "in english": "translate_th_en", "แปลเป็นภาษาอังกฤษ": "translate_th_en",
    "thai food": "find_thai_food", "pad thai": "find_thai_food", "อาหารไทย": "find_thai_food",
    "ต้มยำ": "find_thai_food", "pad see ew": "find_thai_food", "somtum": "find_thai_food",
    
    # ... more keywords as needed ...
}

# --- LLM Generation Configuration ---
# It's strongly recommended to set the API key via environment variable:
# export DEEPSEEK_API_KEY='your_api_key' (Linux/macOS)
# set DEEPSEEK_API_KEY=your_api_key (Windows)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # Or "deepseek-coder"

# Generation parameters
NUM_GENERATIONS_TO_ATTEMPT = 50  # Number of examples to try generating
MAX_RETRIES_LLM = 3  # Maximum retry attempts for LLM API calls
RETRY_DELAY_LLM = 5  # Delay between retries (seconds)
GENERATION_TEMPERATURE = 0.8  # Higher = more creative but potentially less accurate

# Advanced parameters
MAX_TOKENS_PER_GENERATION = 500  # Maximum tokens to generate
ENABLE_CACHING = True  # Whether to use cache for LLM responses
USE_THAI_QUERIES = True  # Whether to include Thai queries in the prompt

# --- Prompting Enhancements ---
ENABLE_FEW_SHOT = True # Enable/disable few-shot examples in prompt
NUM_FEW_SHOT_EXAMPLES = 2 # Number of few-shot examples to include

# --- Negative Sampling ---
ENABLE_NEGATIVE_SAMPLING = True # Enable/disable generation of negative samples
NEGATIVE_SAMPLING_RATIO = 0.15 # Target ratio of negative samples (e.g., 15%)

# --- Validation and Processing Configuration ---
# Whether to perform each of the validation stages
ENABLE_FORMAT_CHECK = True  # Check JSON format and API structure
ENABLE_EXECUTION_CHECK = True  # Simulate API execution
ENABLE_SEMANTIC_CHECK = True  # Check if keywords match API calls
ENABLE_PLAUSIBILITY_CHECK = True # Enable/disable argument value plausibility checks
ENABLE_SEMANTIC_SIMILARITY_CHECK = False # Placeholder: Enable/disable embedding-based semantic check
SEMANTIC_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # Example model if enabled

# Output formats
SAVE_CSV_OUTPUT = True  # Save data in CSV format (in addition to JSONL)
SAVE_SIMPLIFIED_FORMAT = True  # Save simplified version for non-technical users
GENERATE_VISUALIZATIONS = True  # Generate charts for statistics

# --- Output Enhancements ---
ADD_NEEDS_REVIEW_FLAG = True # Add a flag for samples that might need manual review

# --- Validation ---
# Check if API key is available
if not DEEPSEEK_API_KEY:
    logging.warning("DEEPSEEK_API_KEY environment variable not set. LLM generation will fail.")

print(f"APIGen LLM Config Loaded. Output Dir: {SIMULATED_OUTPUT_DIR}")
print(f"Attempting to generate {NUM_GENERATIONS_TO_ATTEMPT} samples.")
print(f"Available APIs: {len(API_LIBRARY)} across various categories")

# Check if output directories exist, create if they don't
for directory in [SIMULATED_OUTPUT_DIR, VISUALIZATIONS_DIR, LOG_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)
