import json
import os
import time
import requests # Added for API calls
import random # Added for random API selection in prompt
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv() # Added

from config_apigen_llm import (
    API_LIBRARY, SIMULATED_OUTPUT_DIR, VERIFIED_DATA_FILENAME,
    SEMANTIC_KEYWORDS_TO_API, DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
    DEEPSEEK_MODEL, NUM_GENERATIONS_TO_ATTEMPT, MAX_RETRIES_LLM,
    RETRY_DELAY_LLM, GENERATION_TEMPERATURE, MAX_TOKENS_PER_GENERATION,
    ENABLE_CACHING, USE_THAI_QUERIES, ENABLE_FORMAT_CHECK,
    ENABLE_EXECUTION_CHECK, ENABLE_SEMANTIC_CHECK, SAVE_CSV_OUTPUT,
    SAVE_SIMPLIFIED_FORMAT, GENERATE_VISUALIZATIONS,
    VISUALIZATIONS_DIR, LOG_DIR, LOG_FILENAME
)
from util_apigen import (
    setup_logger, load_jsonl, save_jsonl, save_csv,
    convert_to_simplified_format, generate_statistics,
    visualize_statistics, check_argument_plausibility,
    check_semantic_similarity
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

# --- Setup Logger ---
logger = setup_logger(os.path.join(LOG_DIR, LOG_FILENAME))

# --- Verification Stages ---
def stage1_format_checker(raw_output):
    """
    Checks if the raw LLM output string is valid JSON and conforms to the expected structure.
    Includes enhanced validation and plausibility checks.
    """
    needs_review = False
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        try:
            # Try extracting JSON from potential markdown blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
            if json_match:
                cleaned_output = json_match.group(1)
            else:
                # Fallback to finding first { and last }
                json_start = raw_output.find('{')
                json_end = raw_output.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    cleaned_output = raw_output[json_start:json_end]
                else:
                    logger.error("Could not find JSON brackets {} in response.")
                    return False, "Invalid JSON (No brackets found)", None, False

            data = json.loads(cleaned_output)
        except (json.JSONDecodeError, TypeError):
            logger.error("Failed to parse JSON even after cleanup attempt.")
            return False, "Invalid JSON format", None, False

    if not isinstance(data, dict):
        return False, "Output is not a JSON object", None, False

    if "query" not in data or "answer" not in data:
        return False, "Missing 'query' or 'answer' field", None, False

    if not isinstance(data["query"], str) or not data["query"]:
        return False, "'query' is not a non-empty string", None, False

    if not isinstance(data["answer"], list):
        return False, "'answer' is not a list", None, False

    validated_calls = []
    plausibility_warnings = []

    for i, call in enumerate(data.get("answer", [])):
        if not isinstance(call, dict):
            return False, f"Call #{i+1} is not an object", None, False

        if "name" not in call or "arguments" not in call:
            return False, f"Call #{i+1} missing 'name' or 'arguments'", None, False

        api_name = call["name"]
        if api_name not in API_LIBRARY:
            return False, f"Call #{i+1} '{api_name}' is not a recognized API", None, False

        # Check required arguments and types
        api_def = API_LIBRARY[api_name]
        for param, details in api_def.get("parameters", {}).items():
            if details.get("required", False) and param not in call.get("arguments", {}):
                return False, f"Call #{i+1} '{api_name}' missing required argument '{param}'", None, False

            if param in call.get("arguments", {}):
                value = call["arguments"][param]
                expected_type = details.get("type")
                if expected_type:
                    # Type validation with automatic conversion where possible
                    try:
                        if expected_type == "string" and not isinstance(value, str):
                            call["arguments"][param] = str(value)
                            needs_review = True
                        elif expected_type == "number":
                            if isinstance(value, str):
                                call["arguments"][param] = float(value)
                                needs_review = True
                        elif expected_type == "integer":
                            if isinstance(value, (str, float)):
                                num = float(value)
                                if num.is_integer():
                                    call["arguments"][param] = int(num)
                                    needs_review = True
                                else:
                                    return False, f"Call #{i+1} '{api_name}' argument '{param}' must be an integer", None, False
                        elif expected_type == "boolean":
                            if isinstance(value, str):
                                if value.lower() == "true":
                                    call["arguments"][param] = True
                                    needs_review = True
                                elif value.lower() == "false":
                                    call["arguments"][param] = False
                                    needs_review = True

                    except (ValueError, TypeError):
                        return False, f"Call #{i+1} '{api_name}' argument '{param}' has invalid type", None, False

        # Plausibility checks
        if ENABLE_PLAUSIBILITY_CHECK:
            is_plausible, issues = check_argument_plausibility(api_name, call["arguments"], API_LIBRARY)
            if issues:
                plausibility_warnings.extend([f"Call #{i+1} '{api_name}': {issue}" for issue in issues])
                if not is_plausible:
                    return False, f"Argument Plausibility Failed: {issues[0]}", None, True
                needs_review = True

        validated_calls.append(call)

    validated_data = {"query": data["query"], "answer": validated_calls}
    reason = "Format OK"
    if plausibility_warnings:
        reason += f" (Warnings: {'; '.join(plausibility_warnings)})"
        logger.warning(f"Plausibility warnings: {'; '.join(plausibility_warnings)}")

    return True, reason, validated_data, needs_review

def stage2_execution_checker(validated_data):
    """Attempts to 'execute' the validated function calls using simulators."""
    query = validated_data.get("query", "")
    calls = validated_data.get("answer", [])
    execution_results = []
    all_calls_succeeded = True
    needs_review = False

    for i, call in enumerate(calls):
        api_name = call.get("name")
        arguments = call.get("arguments", {})
        result = {"call": call}  # Store the original call

        if api_name in API_EXECUTORS:
            try:
                logger.info(f"Executing {api_name} with arguments: {arguments}")
                success, output = API_EXECUTORS[api_name](**arguments)
                result["execution_success"] = success
                result["execution_output"] = output
                if not success:
                    all_calls_succeeded = False
                    needs_review = True
            except Exception as e:
                logger.error(f"Error executing {api_name}: {str(e)}")
                result["execution_success"] = False
                result["execution_output"] = {"error": str(e)}
                all_calls_succeeded = False
                needs_review = True
        else:
            logger.error(f"No executor found for API '{api_name}'")
            result["execution_success"] = False
            result["execution_output"] = {"error": f"No executor found for API '{api_name}'"}
            all_calls_succeeded = False
            needs_review = True

        execution_results.append(result)

    passing_data = {
        "query": query,
        "execution_results": execution_results,
        "needs_review": needs_review
    }
    return all_calls_succeeded, "Execution completed", passing_data

def stage3_semantic_checker(execution_data):
    """
    Performs semantic checks with enhanced validation including semantic similarity
    and expanded keyword matching.
    """
    query = execution_data.get("query", "").lower()
    results = execution_data.get("execution_results", [])
    needs_review = execution_data.get("needs_review", False)

    # Basic keyword matching
    expected_apis = set()
    for keyword, api_name in SEMANTIC_KEYWORDS_TO_API.items():
        if keyword.lower() in query:
            expected_apis.add(api_name)

    # Get successfully executed APIs
    executed_successfully = set()
    for result in results:
        if result.get("execution_success"):
            executed_successfully.add(result["call"]["name"])

    # Enhanced semantic check with embedding similarity
    if ENABLE_SEMANTIC_SIMILARITY_CHECK:
        logger.info("Performing semantic similarity check")
        for result in results:
            api_name = result["call"]["name"]
            api_desc = API_LIBRARY[api_name]["description"]
            similarity = check_semantic_similarity(query, api_desc, SEMANTIC_EMBEDDING_MODEL)
            if similarity < 0.3:  # Threshold can be adjusted
                logger.warning(f"Low semantic similarity ({similarity:.2f}) between query and {api_name}")
                needs_review = True

    # Decision logic
    semantic_ok = True
    reason = "Semantic check completed"

    if expected_apis:
        if not expected_apis.intersection(executed_successfully):
            semantic_ok = False
            reason = f"Query suggests {expected_apis} but executed {executed_successfully}"
            needs_review = True
    elif not executed_successfully:
        # If no specific APIs were expected but none were executed successfully
        semantic_ok = False
        reason = "No successful API executions"
        needs_review = True

    execution_data["needs_review"] = needs_review
    return semantic_ok, reason, execution_data

# --- Main Pipeline ---

from typing import Optional, Callable
from pipeline_utils import PipelineProgress

def run_pipeline(
    progress_callback: Optional[Callable] = None,
    stage_callback: Optional[Callable] = None
):
    """
    Runs the full APIGen pipeline with LLM generation.
    
    Args:
        progress_callback: Optional callback function to report progress (0.0 to 1.0)
        stage_callback: Optional callback function to report current stage
    """
    # Initialize progress tracking
    progress = PipelineProgress(
        NUM_GENERATIONS_TO_ATTEMPT,
        progress_callback=progress_callback,
        stage_callback=stage_callback
    )
    logger.info("Starting APIGen pipeline...")
    
    # Generate data using LLM
    progress.update_stage('generation')
    raw_llm_outputs = generate_with_llm()
    progress.update_stage_progress(1.0)

    if not raw_llm_outputs:
        logger.error("No outputs generated by LLM. Exiting.")
        return

    verified_data_all_stages = []
    stats = {
        "total_attempted": len(raw_llm_outputs),
        "fail_format": 0,
        "fail_execution": 0,
        "fail_semantic": 0,
        "pass": 0,
        "needs_review": 0
    }

    # Process each generated output
    for i, raw_output in enumerate(raw_llm_outputs):
        logger.info(f"\n--- Processing Generated Output #{i+1}/{len(raw_llm_outputs)} ---")
        start_time = time.time()

        # Stage 1: Format Check
        progress.update_stage('format')
        if ENABLE_FORMAT_CHECK:
            format_ok, reason1, validated_data, needs_review = stage1_format_checker(raw_output)
            progress.update_stage_progress((i + 1) / len(raw_llm_outputs))
            logger.info(f"Stage 1 (Format Check): {'PASS' if format_ok else 'FAIL'} - {reason1}")
            if not format_ok:
                stats["fail_format"] += 1
                if needs_review:
                    stats["needs_review"] += 1
                continue
        else:
            try:
                validated_data = json.loads(raw_output)
                needs_review = False
            except json.JSONDecodeError:
                logger.error("Basic JSON parsing failed when format check disabled")
                stats["fail_format"] += 1
                continue

        # Stage 2: Execution Check
        progress.update_stage('execution')
        if ENABLE_EXECUTION_CHECK:
            exec_ok, reason2, execution_data = stage2_execution_checker(validated_data)
            progress.update_stage_progress((i + 1) / len(raw_llm_outputs))
            logger.info(f"Stage 2 (Execution Check): {'PASS' if exec_ok else 'FAIL'} - {reason2}")
            if not exec_ok:
                stats["fail_execution"] += 1
                if execution_data.get("needs_review"):
                    stats["needs_review"] += 1
                continue
        else:
            execution_data = validated_data

        # Stage 3: Semantic Check
        progress.update_stage('semantic')
        if ENABLE_SEMANTIC_CHECK:
            semantic_ok, reason3, final_data = stage3_semantic_checker(execution_data)
            progress.update_stage_progress((i + 1) / len(raw_llm_outputs))
            logger.info(f"Stage 3 (Semantic Check): {'PASS' if semantic_ok else 'FAIL'} - {reason3}")
            if not semantic_ok:
                stats["fail_semantic"] += 1
                if final_data.get("needs_review"):
                    stats["needs_review"] += 1
                continue
        else:
            final_data = execution_data

        # Track needs_review flag
        if final_data.get("needs_review", False) or needs_review:
            stats["needs_review"] += 1
            final_data["needs_review"] = True

        # If all enabled stages passed
        stats["pass"] += 1
        verified_data_all_stages.append(final_data)
        progress.increment_sample()
        duration = time.time() - start_time
        logger.info(f"--- PASSED ALL STAGES ({duration:.2f}s) ---")

    logger.info("\n=== Pipeline Finished ===")
    logger.info("Verification Statistics:")
    for key, value in stats.items():
        logger.info(f"- {key.replace('_', ' ').title()}: {value}")

    # Save results if any data passed verification
    if verified_data_all_stages:
        # Save verified data in JSONL format
        output_path = os.path.join(SIMULATED_OUTPUT_DIR, VERIFIED_DATA_FILENAME)
        if save_jsonl(verified_data_all_stages, output_path):
            logger.info(f"Saved {stats['pass']} verified examples to: {output_path}")
        
        # Save CSV format if enabled
        if SAVE_CSV_OUTPUT:
            csv_path = os.path.join(SIMULATED_OUTPUT_DIR, VERIFIED_DATA_CSV_FILENAME)
            if save_csv(verified_data_all_stages, csv_path):
                logger.info(f"Saved CSV version to: {csv_path}")

        # Save simplified format if enabled
        if SAVE_SIMPLIFIED_FORMAT:
            simplified_data = convert_to_simplified_format(verified_data_all_stages)
            simplified_path = os.path.join(SIMULATED_OUTPUT_DIR, SIMPLIFIED_DATA_FILENAME)
            if save_jsonl(simplified_data, simplified_path):
                logger.info(f"Saved simplified format to: {simplified_path}")

        # Generate statistics and visualizations
        stats_data = generate_statistics(verified_data_all_stages)
        if stats_data:
            stats_path = os.path.join(SIMULATED_OUTPUT_DIR, "api_call_statistics.json")
            try:
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved detailed statistics to: {stats_path}")
            
                if GENERATE_VISUALIZATIONS:
                    visualize_statistics(stats_data, VISUALIZATIONS_DIR)
            except Exception as e:
                logger.error(f"Error saving statistics: {e}")
    else:
        logger.warning("No data passed all verification stages.")

if __name__ == "__main__":
    run_pipeline()
