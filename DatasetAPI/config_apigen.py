"""
Configuration file for APIGen Simulation.
Contains definitions of available APIs.
"""
import os # Added for environment variable access

# Import the API library and configurations from config_apigen_llm
from config_apigen_llm import (
    API_LIBRARY, SIMULATED_OUTPUT_DIR, VERIFIED_DATA_FILENAME,
    SEMANTIC_KEYWORDS_TO_API, DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
    DEEPSEEK_MODEL, NUM_GENERATIONS_TO_ATTEMPT, MAX_RETRIES_LLM,
    RETRY_DELAY_LLM, GENERATION_TEMPERATURE
)

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
