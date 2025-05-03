"""
Simulator for API execution logic.
These functions mimic calling real APIs.
"""
import random

def execute_get_weather(location, unit="Celsius"):
    """Simulates fetching weather."""
    if not isinstance(location, str) or not location:
        return False, {"error": "Invalid location parameter"}
    if unit not in ["Celsius", "Fahrenheit"]:
        unit = "Celsius" # Default fallback

    # Simulate potential API failures
    if random.random() < 0.05: # 5% chance of failure
        return False, {"error": "Simulated API timeout"}

    temp = random.uniform(-10, 40) if unit == "Celsius" else random.uniform(14, 104)
    condition = random.choice(["Sunny", "Cloudy", "Rainy", "Snowy"]) # Simplified
    return True, {"location": location, "temperature": round(temp, 1), "unit": unit, "condition": condition}

def execute_search_news(topic, max_results=5):
    """Simulates searching for news."""
    if not isinstance(topic, str) or not topic:
        return False, {"error": "Invalid topic parameter"}
    try:
        num_results = int(max_results)
        if num_results <= 0:
            num_results = 1
    except (ValueError, TypeError):
        num_results = 5 # Default fallback

    # Simulate potential API failures
    if random.random() < 0.1: # 10% chance of failure
        return False, {"error": "Simulated search error"}

    headlines = [f"Article {i+1} about {topic}" for i in range(min(num_results, random.randint(1, 10)))] # Simulate finding 1 to min(max, 10) articles
    return True, {"topic": topic, "headlines": headlines}

def execute_send_message(recipient, message_body):
    """Simulates sending a message."""
    if not isinstance(recipient, str) or not recipient:
        return False, {"error": "Invalid recipient parameter"}
    if not isinstance(message_body, str) or not message_body:
        return False, {"error": "Invalid message_body parameter"}

    # Simulate potential API failures
    if random.random() < 0.02: # 2% chance of failure
         return False, {"error": "Simulated message delivery failure"}

    return True, {"status": "Message sent", "recipient": recipient, "preview": message_body[:30] + "..."}

# Map API names (as strings) to the actual simulation functions
API_EXECUTORS = {
    "get_weather": execute_get_weather,
    "search_news": execute_search_news,
    "send_message": execute_send_message,
}
