import os
import json
import requests
import yfinance as yf
import pytesseract
from PIL import Image
from pythainlp import sent_tokenize, word_tokenize

def get_current_weather(location, format):
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        return json.dumps({"error": "Weather API key not configured"})
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric" if format == "celsius" else "imperial"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        return json.dumps({format: temperature})
    except requests.RequestException as e:
        return json.dumps({"error": str(e)})

def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='1d')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            return json.dumps({"symbol": symbol, "price": current_price})
        return json.dumps({"error": "No data found for symbol"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def perform_ocr(image_path, lang='tha'):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
        sentences = sent_tokenize(text, engine='whitespace+newline')
        words = [word_tokenize(sentence, engine='newmm') for sentence in sentences]
        return json.dumps({
            "text": text,
            "sentences": sentences,
            "words": words
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "format": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit to use"}
                },
                "required": ["location", "format"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The stock symbol, e.g. AAPL"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_ocr",
            "description": "Perform OCR on an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image file"}
                },
                "required": ["image_path"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
    "perform_ocr": perform_ocr
}