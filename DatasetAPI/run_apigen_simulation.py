import json
import os
import time
import requests # Added for API calls
import random # Added for random API selection in prompt
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv() # Added

from config_apigen import (
    API_LIBRARY, SIMULATED_OUTPUT_DIR, VERIFIED_DATA_FILENAME,
    SEMANTIC_KEYWORDS_TO_API, DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
    DEEPSEEK_MODEL, NUM_GENERATIONS_TO_ATTEMPT, MAX_RETRIES_LLM,
    RETRY_DELAY_LLM, GENERATION_TEMPERATURE
)
from api_execution_sim import API_EXECUTORS

# --- LLM Output Generation (Replaces Simulation) ---

def format_apis_for_prompt(num_apis=2):
    """Formats a random subset of API definitions for the LLM prompt."""
    available_apis = list(API_LIBRARY.keys())
    if len(available_apis) <= num_apis:
        selected_api_names = available_apis
    else:
        selected_api_names = random.sample(available_apis, num_apis)

    prompt_str = "Available APIs:\n"
    for name in selected_api_names:
        details = API_LIBRARY[name]
        prompt_str += f"- Function: {name}\n"
        prompt_str += f"  Description: {details['description']}\n"
        prompt_str += f"  Parameters:\n"
        if not details['parameters']:
            prompt_str += "    None\n"
        else:
            for param, p_details in details['parameters'].items():
                req = "(required)" if p_details.get('required') else "(optional)"
                dtype = p_details.get('type', 'any')
                desc = p_details.get('description', '')
                default = f", default={p_details['default']}" if 'default' in p_details else ""
                prompt_str += f"    - {param} ({dtype}) {req}: {desc}{default}\n"
    prompt_str += "\n"
    return prompt_str

def create_llm_prompt():
    """Creates the prompt for the LLM to generate query-answer pairs."""
    api_definitions = format_apis_for_prompt(random.randint(1, 3)) # Use 1 to 3 random APIs

    prompt = f"""You are an AI assistant designed to generate data for training function-calling models.
Your task is to generate a user query and the corresponding API call(s) needed to fulfill that query, based ONLY on the provided API definitions.
Output MUST be a single JSON object containing a "query" field (string) and an "answer" field (a list of JSON objects, where each object has "name" and "arguments" fields).

{api_definitions}

Example Output Format:
{{
  "query": "User's request here",
  "answer": [
    {{
      "name": "api_function_name",
      "arguments": {{
        "param1": "value1",
        "param2": value2
      }}
    }}
    // , {{...}} // Optional: for multiple calls if needed by the query
  ]
}}

Generate a new, diverse example based on the APIs provided above. Ensure the arguments match the API definition.
Generate ONLY the JSON object. Do not include any other text before or after the JSON.
"""
    return prompt

def generate_with_llm():
    """Generates query-answer pairs by calling the DeepSeek LLM."""
    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set. Cannot call LLM.")
        return [] # Return empty list if no key

    generated_outputs = []
    print(f"Attempting to generate {NUM_GENERATIONS_TO_ATTEMPT} examples using {DEEPSEEK_MODEL}...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    for i in range(NUM_GENERATIONS_TO_ATTEMPT):
        print(f"  Generating example {i+1}/{NUM_GENERATIONS_TO_ATTEMPT}...")
        prompt = create_llm_prompt()
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                # {"role": "system", "content": "You generate JSON data for function calling."}, # Optional system message
                {"role": "user", "content": prompt}
            ],
            "temperature": GENERATION_TEMPERATURE,
            "max_tokens": 500 # Adjust as needed
        }

        for attempt in range(MAX_RETRIES_LLM):
            try:
                response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                result = response.json()
                llm_output = result['choices'][0]['message']['content']

                # Basic cleanup: Try to extract only the JSON part if the LLM added extra text
                json_start = llm_output.find('{')
                json_end = llm_output.rfind('}')
                if json_start != -1 and json_end != -1:
                    extracted_json = llm_output[json_start:json_end+1]
                    generated_outputs.append(extracted_json)
                    print(f"    Success (Attempt {attempt+1}).")
                    break # Success, move to next generation
                else:
                    print(f"    Warning (Attempt {attempt+1}): Could not extract JSON from LLM output: {llm_output[:100]}...")
                    if attempt == MAX_RETRIES_LLM - 1:
                         print(f"    Failed to get valid JSON after {MAX_RETRIES_LLM} attempts.")
                    else:
                         time.sleep(RETRY_DELAY_LLM) # Wait before retrying

            except requests.exceptions.RequestException as e:
                print(f"    Error (Attempt {attempt+1}): API request failed: {e}")
                if attempt < MAX_RETRIES_LLM - 1:
                    print(f"    Retrying in {RETRY_DELAY_LLM} seconds...")
                    time.sleep(RETRY_DELAY_LLM)
                else:
                    print(f"    Failed after {MAX_RETRIES_LLM} attempts.")
            except Exception as e:
                print(f"    Error (Attempt {attempt+1}): Unexpected error: {e}")
                # Decide if retry makes sense for unexpected errors
                if attempt < MAX_RETRIES_LLM - 1:
                     time.sleep(RETRY_DELAY_LLM)
                else:
                     print(f"    Failed after {MAX_RETRIES_LLM} attempts.")

    print(f"Finished LLM generation. Got {len(generated_outputs)} raw outputs.")
    return generated_outputs


# --- Verification Stages ---

def stage1_format_checker(raw_output):
    """Checks if the raw LLM output string is valid JSON and conforms to the expected structure."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return False, "Invalid JSON", None

    if not isinstance(data, dict):
        return False, "Output is not a JSON object", None

    if "query" not in data or "answer" not in data:
        return False, "Missing 'query' or 'answer' field", None

    if not isinstance(data["query"], str) or not data["query"]:
        return False, "'query' is not a non-empty string", None

    if not isinstance(data["answer"], list):
        return False, "'answer' is not a list", None

    validated_calls = []
    for i, call in enumerate(data["answer"]):
        if not isinstance(call, dict):
            return False, f"Call #{i+1} in 'answer' is not an object", None
        if "name" not in call or "arguments" not in call:
            return False, f"Call #{i+1} missing 'name' or 'arguments'", None
        if not isinstance(call["name"], str) or not call["name"]:
            return False, f"Call #{i+1} 'name' is not a non-empty string", None
        if not isinstance(call["arguments"], dict):
            return False, f"Call #{i+1} 'arguments' is not an object", None

        # Check if API exists in library
        api_name = call["name"]
        if api_name not in API_LIBRARY:
            return False, f"Call #{i+1} '{api_name}' is not a recognized API", data

        # Check for required arguments
        api_def = API_LIBRARY[api_name]
        for param, details in api_def["parameters"].items():
            if details.get("required", False) and param not in call["arguments"]:
                return False, f"Call #{i+1} '{api_name}' missing required argument '{param}'", data

        # (Optional) Simple type check could be added here based on API_LIBRARY defs
        validated_calls.append(call) # Keep original call structure if valid so far

    # If all checks pass for all calls
    # Return the validated structured data (original query + list of validated calls)
    validated_data = {"query": data["query"], "answer": validated_calls}
    return True, "Format OK", validated_data


def stage2_execution_checker(validated_data):
    """Attempts to 'execute' the validated function calls using simulators."""
    query = validated_data["query"]
    calls = validated_data["answer"]
    execution_results = []
    all_calls_succeeded = True

    for i, call in enumerate(calls):
        api_name = call["name"]
        arguments = call["arguments"]
        executor = API_EXECUTORS.get(api_name)

        if not executor:
            # Should not happen if format checker worked, but double-check
            result = (False, {"error": f"No executor found for API '{api_name}'"})
            all_calls_succeeded = False
        else:
            try:
                # Execute the corresponding simulation function
                print(f"  Executing call {i+1}: {api_name}({arguments})")
                success, exec_output = executor(**arguments)
                result = (success, exec_output)
                if not success:
                    all_calls_succeeded = False
            except TypeError as e: # Handles case where arguments don't match function signature
                 result = (False, {"error": f"Execution error (TypeError) for {api_name}: {e}"})
                 all_calls_succeeded = False
            except Exception as e:
                 result = (False, {"error": f"Unexpected execution error for {api_name}: {e}"})
                 all_calls_succeeded = False

        execution_results.append({
            "call": call,
            "execution_success": result[0],
            "execution_output": result[1]
        })


    passing_data = {
        "query": query,
        "execution_results": execution_results # Contains original call + success status + output/error
    }
    return all_calls_succeeded, "Execution attempted", passing_data


def stage3_semantic_checker(execution_data):
    """Performs a *very basic* semantic check."""
    query = execution_data["query"].lower()
    results = execution_data["execution_results"]

    # Basic Check: If query contains keywords related to an API,
    # at least one *successful* execution should involve that API.
    # This is a huge simplification. Real semantic checks are complex.

    expected_apis = set()
    for keyword, api_name in SEMANTIC_KEYWORDS_TO_API.items():
        if keyword in query:
            expected_apis.add(api_name)

    if not expected_apis: # If no keywords found, maybe pass? Or fail? Let's pass for now.
        return True, "Passed (No specific keywords detected)", execution_data

    executed_successfully = set()
    for result in results:
        if result["execution_success"]:
            executed_successfully.add(result["call"]["name"])

    # Check if at least one expected API was successfully executed
    if expected_apis.intersection(executed_successfully):
        return True, "Passed (Basic keyword match)", execution_data
    else:
        fail_reason = f"Failed: Query keywords suggest {expected_apis}, " \
                      f"but successful executions were {executed_successfully or 'None'}"
        return False, fail_reason, execution_data

# --- Main Pipeline Orchestration ---

def run_pipeline():
    """Runs the full APIGen pipeline with LLM generation."""
    # Replace simulation call with LLM call
    raw_llm_outputs = generate_with_llm() # Changed function call

    if not raw_llm_outputs:
        print("\nNo outputs generated by LLM. Exiting.")
        return

    verified_data_all_stages = []
    stats = {"total_attempted": len(raw_llm_outputs), "fail_format": 0, "fail_execution": 0, "fail_semantic": 0, "pass": 0}

    # Process the raw outputs from the LLM
    for i, raw_output in enumerate(raw_llm_outputs):
        print(f"\n--- Processing Generated Output #{i+1} ---")
        # stats["total"] += 1 # Renamed stat
        start_time = time.time()

        # Stage 1
        format_ok, reason1, validated_data = stage1_format_checker(raw_output)
        print(f"Stage 1 (Format Check): {'PASS' if format_ok else 'FAIL'} - {reason1}")
        if not format_ok:
            stats["fail_format"] += 1
            # Optionally log the raw_output that failed format check
            # print(f"      Raw Output: {raw_output}")
            continue

        # Stage 2
        # ... (rest of the pipeline logic remains the same) ...
        exec_ok, reason2, execution_data = stage2_execution_checker(validated_data)
        print(f"Stage 2 (Execution Check): {'ALL PASS' if exec_ok else 'SOME FAIL'} - {reason2}")
        if not exec_ok:
            stats["fail_execution"] += 1
            # print(f"      Details: {execution_data['execution_results']}") # To see errors
            continue # Skip Semantic if execution had errors

        # Stage 3
        semantic_ok, reason3, final_data = stage3_semantic_checker(execution_data)
        print(f"Stage 3 (Semantic Check): {'PASS' if semantic_ok else 'FAIL'} - {reason3}")
        if not semantic_ok:
            stats["fail_semantic"] += 1
            continue

        # If all stages passed
        stats["pass"] += 1
        verified_data_all_stages.append(final_data)
        duration = time.time() - start_time
        print(f"--- PASSED ALL STAGES ({duration:.2f}s) ---")

    print("\n--- Pipeline Finished ---")
    print("Verification Statistics:")
    for key, value in stats.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")

    # Save verified data
    if verified_data_all_stages:
        # Use BASE_PATH from config for correct output path construction
        output_path = os.path.join(SIMULATED_OUTPUT_DIR, VERIFIED_DATA_FILENAME)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Use dirname for makedirs
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in verified_data_all_stages:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"\nSaved {stats['pass']} verified data points to: {output_path}")
        except Exception as e:
            print(f"\nError saving verified data: {e}")
    else:
        print("\nNo data passed all verification stages.")

if __name__ == "__main__":
    run_pipeline()
