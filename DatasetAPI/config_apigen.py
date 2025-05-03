"""
Configuration file for APIGen Simulation.
Contains definitions of available APIs.
"""
import os # Added for environment variable access

API_LIBRARY = {
    "get_weather": {
        "description": "Get the current weather for a specific location.",
        "parameters": {
            "location": {"type": "string", "description": "The city name", "required": True},
            "unit": {"type": "string", "description": "Temperature unit (Celsius or Fahrenheit)", "required": False, "default": "Celsius"}
        }
    },
    "search_news": {
        "description": "Search for recent news articles on a given topic.",
        "parameters": {
            "topic": {"type": "string", "description": "The topic to search news for", "required": True},
            "max_results": {"type": "integer", "description": "Maximum number of news articles to return", "required": False, "default": 5}
        }
    },
    "send_message": {
        "description": "Send a message to a recipient.",
        "parameters": {
            "recipient": {"type": "string", "description": "The name or ID of the recipient", "required": True},
            "message_body": {"type": "string", "description": "The content of the message", "required": True}
        }
    },
    # Add more simulated APIs here if needed
}

# --- Output Configuration ---
# Determine base path assuming this file is in DatasetAPI
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMULATED_OUTPUT_DIR = os.path.join(BASE_PATH, "DataOutput", "apigen_real_llm") # Changed output subfolder
VERIFIED_DATA_FILENAME = "verified_api_calls_llm.jsonl"

# --- Semantic Checker Keywords (Very Basic) ---
# Map keywords in query to expected API function names for basic semantic check
SEMANTIC_KEYWORDS_TO_API = {
    "weather": "get_weather",
    "temperature": "get_weather",
    "forecast": "get_weather",
    "news": "search_news",
    "article": "search_news",
    "headline": "search_news",
    "send": "send_message",
    "message": "send_message",
    "text": "send_message",
}

# --- LLM Generation Configuration ---
# It's strongly recommended to set the API key via environment variable:
# export DEEPSEEK_API_KEY='your_api_key'
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat" # Or "deepseek-coder"
NUM_GENERATIONS_TO_ATTEMPT = 10 # How many examples to try generating
MAX_RETRIES_LLM = 2
RETRY_DELAY_LLM = 5 # seconds
GENERATION_TEMPERATURE = 0.7 # For diversity
