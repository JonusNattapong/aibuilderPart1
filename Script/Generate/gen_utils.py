import re
import json
import time
from config_generate import MAX_RETRIES, RETRY_DELAY

def parse_json_output(llm_output):
    """Attempts to parse JSON from the LLM output, handling markdown code blocks."""
    if not llm_output:
        return None
    try:
        # Look for JSON within ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            # Basic cleaning for common issues like trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str)
        else:
            # Fallback: try parsing the whole output if no code block found
            # Basic cleaning for common issues like trailing commas
            cleaned_output = re.sub(r',\s*([}\]])', r'\1', llm_output)
            return json.loads(cleaned_output)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from LLM output: {llm_output}")
        return None
    except Exception as e:
        print(f"Error parsing JSON: {e}\nOutput: {llm_output}")
        return None

def invoke_llm_with_retry(chain, inputs, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Invokes the LLM chain with retry logic."""
    for attempt in range(max_retries):
        try:
            response = chain.invoke(inputs)
            # Check if response is a dictionary and has the expected key (often 'text')
            if isinstance(response, dict) and 'text' in response:
                 return response['text']
            elif isinstance(response, str): # Handle direct string output if chain configured differently
                 return response
            else:
                 print(f"Warning: Unexpected response format: {response}")
                 return None # Or handle appropriately
        except Exception as e:
            print(f"LLM invocation failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this item.")
                return None
    return None # Should not be reached if max_retries > 0, but added for safety
