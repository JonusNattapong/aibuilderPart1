import os
import json
import time
import logging
import requests # Added
import random   # Added
import hashlib  # Added
from utils import save_jsonl, save_csv, convert_to_simplified_format, generate_statistics, visualize_statistics
from config import (
    ENABLE_EXECUTION_CHECK, ENABLE_SEMANTIC_CHECK, SAVE_CSV_OUTPUT, SAVE_SIMPLIFIED_FORMAT,
    GENERATE_VISUALIZATIONS, VERIFIED_DATA_CSV_FILENAME, SIMPLIFIED_DATA_FILENAME,
    STATISTICS_FILENAME, VISUALIZATIONS_DIR, output_dir, output_path
)
from stage1_format_checker import stage1_format_checker
from stage2_execution_checker import stage2_execution_checker
from stage3_semantic_checker import stage3_semantic_checker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
Main script for generating API call dataset using LLM.
This runs the complete pipeline: LLM generation, validation, and data processing.
"""
# ... (imports remain largely the same) ...
import re # Added for plausibility checks via util

# Import configurations and utilities
from config_apigen_llm import (
    API_LIBRARY, SIMULATED_OUTPUT_DIR, VERIFIED_DATA_FILENAME,
    VERIFIED_DATA_CSV_FILENAME, SIMPLIFIED_DATA_FILENAME,
    SEMANTIC_KEYWORDS_TO_API, DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
    DEEPSEEK_MODEL, NUM_GENERATIONS_TO_ATTEMPT, MAX_RETRIES_LLM,
    RETRY_DELAY_LLM, GENERATION_TEMPERATURE, MAX_TOKENS_PER_GENERATION,
    ENABLE_CACHING, USE_THAI_QUERIES, ENABLE_FORMAT_CHECK,
    ENABLE_EXECUTION_CHECK, ENABLE_SEMANTIC_CHECK, SAVE_CSV_OUTPUT,
    SAVE_SIMPLIFIED_FORMAT, GENERATE_VISUALIZATIONS,
    VISUALIZATIONS_DIR, LOG_DIR, LOG_FILENAME, CACHE_DIR, CACHE_FILENAME,
    STATISTICS_FILENAME, FEW_SHOT_EXAMPLES_FILENAME, # Added
    ENABLE_FEW_SHOT, NUM_FEW_SHOT_EXAMPLES, # Added
    ENABLE_NEGATIVE_SAMPLING, NEGATIVE_SAMPLING_RATIO, # Added
    ENABLE_PLAUSIBILITY_CHECK, ADD_NEEDS_REVIEW_FLAG, # Added
    ENABLE_SEMANTIC_SIMILARITY_CHECK, SEMANTIC_EMBEDDING_MODEL # Added
)
from api_execution_sim_llm import API_EXECUTORS
from util_apigen import (
    setup_logger, load_jsonl, save_jsonl, save_csv, convert_to_simplified_format,
    generate_statistics, visualize_statistics, LLMResponseCache,
    check_argument_plausibility, check_semantic_similarity # Added
)

# --- Setup Logger ---
logger = setup_logger(os.path.join(LOG_DIR, LOG_FILENAME))

# --- Initialize Cache if enabled ---
# ... existing code ...

# --- Load Few-Shot Examples if enabled ---
few_shot_examples = []
if ENABLE_FEW_SHOT:
    few_shot_path = os.path.join(SIMULATED_OUTPUT_DIR, FEW_SHOT_EXAMPLES_FILENAME)
    if os.path.exists(few_shot_path):
        all_verified = load_jsonl(few_shot_path)
        # Simple random selection for now
        if len(all_verified) >= NUM_FEW_SHOT_EXAMPLES:
            few_shot_examples = random.sample(all_verified, NUM_FEW_SHOT_EXAMPLES)
            logger.info(f"Loaded {len(few_shot_examples)} few-shot examples from {few_shot_path}")
        else:
            logger.warning(f"Not enough verified examples ({len(all_verified)}) in {few_shot_path} for {NUM_FEW_SHOT_EXAMPLES} few-shot examples.")
    else:
        logger.warning(f"Few-shot examples file not found: {few_shot_path}. Few-shot disabled.")
        ENABLE_FEW_SHOT = False # Disable if file not found

# --- Command-line arguments ---
# ... existing code ...

# --- LLM Prompt Functions ---

def format_apis_for_prompt(num_apis_min=3, num_apis_max=6):
    """Formats a random subset of API definitions for the LLM prompt."""
    available_apis = list(API_LIBRARY.keys())
    num_apis = random.randint(num_apis_min, min(num_apis_max, len(available_apis)))
    if len(available_apis) <= num_apis:
        selected_api_names = available_apis
    else:
        selected_api_names = random.sample(available_apis, num_apis)

    # Format the selected APIs as a string for the prompt
    prompt_str = "Available APIs:\n---\n" # Added separator
    for name in selected_api_names:
        details = API_LIBRARY[name]
        prompt_str += f"- Function: {name}\n"
        prompt_str += f"  Description: {details['description']}\n"
        prompt_str += f"  Category: {details.get('category', 'N/A')}\n" # Added category display
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
        prompt_str += "\n" # Add a newline after each API definition

    prompt_str += "---\n" # Added separator
    return prompt_str, selected_api_names

def format_few_shot_examples(examples: List[Dict]) -> str:
    """Formats few-shot examples for the prompt."""
    if not examples:
        return ""
    
    formatted = "Here are some examples of desired input/output:\n"
    for ex in examples:
        query = ex.get("query")
        # Reconstruct the 'answer' part from 'execution_results'
        answer = [res.get("call") for res in ex.get("execution_results", []) if res.get("call")]
        if query and answer is not None: # Ensure both exist
             formatted += f"\nExample Input Query: {query}\n"
             formatted += f"Example Output JSON:\n```json\n"
             formatted += json.dumps({"query": query, "answer": answer}, indent=2, ensure_ascii=False)
             formatted += "\n```\n"
    formatted += "---\n"
    return formatted

def create_llm_prompt(use_thai=False, generate_negative=False):
    """Creates the prompt for the LLM to generate query-answer pairs."""
    api_definitions, selected_apis = format_apis_for_prompt()
    few_shot_str = format_few_shot_examples(few_shot_examples) if ENABLE_FEW_SHOT else ""

    # Basic system prompt
    system_prompt = """You are an AI assistant designed to generate data for training function-calling models.
Your task is to generate a user query and the corresponding API call(s) needed to fulfill that query, based ONLY on the provided API definitions.
Output MUST be a single JSON object containing a "query" field (string) and an "answer" field (a list of JSON objects, where each object has "name" and "arguments" fields).
"""
    if generate_negative:
        system_prompt += "\nIMPORTANT: For this specific request, generate a user query that DOES NOT require any of the provided API calls. The 'answer' field in the JSON output MUST be an empty list []."
    elif use_thai:
        system_prompt += "\nOccasionally (~30% of the time), generate Thai language queries instead of English."

    # User prompt with API definitions and examples
    user_prompt = f"""
{api_definitions}
{few_shot_str}
Output Format Reminder:
{{
  "query": "User's request here",
  "answer": [
    {{
      "name": "api_function_name",
      "arguments": {{ ... }}
    }}
    // , {{...}} // Optional: for multiple calls
  ]
  // OR an empty list [] if no API call is needed (especially if asked to generate such a case)
}}

Generate a new, diverse example based ONLY on the APIs provided above.
The query should be realistic and natural.
Ensure the arguments match the API definition exactly.
Generate ONLY the JSON object. Do not include any other text before or after the JSON.
"""
    # Add Thai examples if needed (same as before)
    # ...

    return system_prompt, user_prompt, selected_apis

def generate_with_llm(num_samples=None, temperature=None):
    """Generates query-answer pairs by calling the DeepSeek LLM."""
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY is not set in config_apigen_llm.py")
        return []
    if not DEEPSEEK_API_URL:
        logger.error("DEEPSEEK_API_URL is not set in config_apigen_llm.py")
        return []

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    cache = LLMResponseCache(os.path.join(CACHE_DIR, CACHE_FILENAME)) if ENABLE_CACHING else None
    if ENABLE_CACHING and not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

    generated_outputs = []
    successful_generations = 0
    attempted_generations = 0

    logger.info(f"Attempting to generate approximately {num_samples} samples (including negative samples if enabled)...")

    while successful_generations < num_samples:
        attempted_generations += 1
        generate_negative = ENABLE_NEGATIVE_SAMPLING and random.random() < NEGATIVE_SAMPLING_RATIO
        system_prompt, user_prompt, selected_apis = create_llm_prompt(USE_THAI_QUERIES, generate_negative)

        cache_key = hashlib.md5((system_prompt + user_prompt + str(temperature) + DEEPSEEK_MODEL).encode()).hexdigest()

        if ENABLE_CACHING and cache and cache.get(cache_key):
            logger.info(f"  Cache HIT for attempt {attempted_generations}. Using cached response.")
            raw_output = cache.get(cache_key)
            try:
                # Attempt to parse the cached output, which should be the raw LLM string
                json_start = raw_output.find('{')
                json_end = raw_output.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    cleaned_output = raw_output[json_start:json_end]
                    parsed_json = json.loads(cleaned_output)
                else: # Fallback if no brackets, assume it might be full JSON string
                    parsed_json = json.loads(raw_output)

                parsed_json['_generation_meta'] = {'is_negative_sample_target': generate_negative}
                generated_outputs.append(json.dumps(parsed_json))
                successful_generations += 1
                logger.info(f"    Success (from cache) ({successful_generations}/{num_samples})")
                continue
            except (json.JSONDecodeError, TypeError) as e: # Added TypeError for safety
                logger.warning(f"    Warning: Could not parse cached JSON: {e}. Will regenerate.")
                if cache:
                    cache.delete(cache_key)

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": MAX_TOKENS_PER_GENERATION
        }

        retries = 0
        while retries <= MAX_RETRIES_LLM:
            try:
                logger.info(f"  Attempt {attempted_generations}, Target {successful_generations+1}/{num_samples}, "
                             f"Type: {'Negative' if generate_negative else 'Positive'}, Retry {retries}...")
                
                response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
                response.raise_for_status()

                result = response.json()
                raw_output = result['choices'][0]['message']['content']

                try:
                    json_start = raw_output.find('{')
                    json_end = raw_output.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        cleaned_output = raw_output[json_start:json_end]
                        parsed_json = json.loads(cleaned_output)
                        parsed_json['_generation_meta'] = {'is_negative_sample_target': generate_negative}
                        generated_outputs.append(json.dumps(parsed_json))
                        successful_generations += 1

                        if ENABLE_CACHING and cache:
                            cache.set(cache_key, raw_output) # Cache the raw output

                        logger.info(f"    Success ({successful_generations}/{num_samples})")
                        break 
                    else:
                         logger.warning(f"    Warning: Could not find JSON brackets {{}} in LLM response.")
                except (ValueError, json.JSONDecodeError) as json_err:
                    logger.warning(f"    Warning: Could not extract valid JSON from LLM response: {json_err}")
                    logger.debug(f"      Raw Response: {raw_output}")
            except requests.exceptions.RequestException as e:
                logger.error(f"    Error making request to Deepseek API: {e}")
                if retries >= MAX_RETRIES_LLM:
                    logger.error("    Max retries reached for API call")
                    break 
            retries += 1 # Ensure no 'ฝ' character here
            time.sleep(RETRY_DELAY_LLM)

        if successful_generations == num_samples: # Check if we've met the target
            break
        # Safety break if something goes wrong and loop runs too long
        if attempted_generations > num_samples * (MAX_RETRIES_LLM + 2) and successful_generations < num_samples : # Adjusted safety break
             logger.warning(f"Breaking generation loop early after {attempted_generations} attempts for {num_samples} targets.")
             break

    logger.info(f"\nLLM Generation finished. Successfully generated {successful_generations} raw outputs from {attempted_generations} attempts.")
    if ENABLE_CACHING and cache:
        logger.info(f"Cache stats: Hits={cache.hits}, Misses={cache.misses}, Sets={cache.sets}")
    return generated_outputs


# --- Verification Stages ---

def stage1_format_checker(raw_output):
    """Checks if the raw LLM output string is valid JSON and conforms to the expected structure. Includes plausibility checks."""
    needs_review = False
    try:
        # Try loading the raw output directly first
        data = json.loads(raw_output)
        # Extract generation metadata if present
        generation_meta = data.pop('_generation_meta', {})
    except json.JSONDecodeError:
        # If direct loading fails, try extracting from potential markdown code blocks
        logger.warning("Initial JSON parsing failed, attempting extraction from markdown.")
        try:
            # Attempt to find JSON within ```json ... ``` or just { ... }
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_output, re.DOTALL | re.IGNORECASE)
            if json_match:
                cleaned_output = json_match.group(1)
            else:
                # Fallback to finding the first '{' and last '}'
                json_start = raw_output.find('{')
                json_end = raw_output.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    cleaned_output = raw_output[json_start:json_end]
                else:
                    logger.error("Could not find JSON brackets {} in LLM response after initial parse failure.")
                    return False, "Invalid JSON (No brackets found)", None, False, {}

            # Basic cleaning for common issues like trailing commas before parsing
            cleaned_output = re.sub(r',\s*([}\]])', r'\1', cleaned_output)
            data = json.loads(cleaned_output)
            generation_meta = data.pop('_generation_meta', {}) # Extract meta again
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON even after extraction attempt: {raw_output[:200]}...")
            return False, "Invalid JSON (Extraction failed)", None, False, {} # Added needs_review, meta

    # --- Continue with existing format checks ---
    if not isinstance(data, dict):
        return False, "Output is not a JSON object", None, False, {}

    if "query" not in data or "answer" not in data:
        return False, "Missing 'query' or 'answer' field", None, False, {}

    if not isinstance(data["query"], str) or not data["query"]:
        # Allow empty query? Maybe flag for review. For now, require non-empty.
        return False, "'query' is not a non-empty string", None, False, {}

    if not isinstance(data["answer"], list):
        # Allow empty list for negative samples
        if data["answer"] == [] and generation_meta.get('is_negative_sample_target', False):
             logger.info("Format OK (Empty answer list for targeted negative sample)")
             return True, "Format OK (Negative Sample)", data, needs_review, generation_meta
        return False, "'answer' is not a list", None, False, {}

    validated_calls = []
    plausibility_warnings = []
    for i, call in enumerate(data.get("answer", [])):
        # ... (existing checks for call structure, name, arguments) ...

        # Check if API exists in library
        api_name = call["name"]
        if api_name not in API_LIBRARY:
            return False, f"Call #{i+1} '{api_name}' is not a recognized API", data, False, {}

        # Check for required arguments
        api_def = API_LIBRARY.get(api_name)
        if not api_def: # Should not happen if previous check passed, but good practice
             return False, f"API definition not found for '{api_name}'", data, False, {}
        required_params = {p for p, d in api_def.get("parameters", {}).items() if d.get("required")}
        provided_args = set(call.get("arguments", {}).keys())
        missing_params = required_params - provided_args
        if missing_params:
            return False, f"Call #{i+1} '{api_name}' missing required argument(s): {', '.join(missing_params)}", data, False, {}

        # Basic type checking (optional but recommended)
        for param, value in call.get("arguments", {}).items():
             param_def = api_def.get("parameters", {}).get(param)
             if param_def:
                 expected_type_str = param_def.get("type")
                 if expected_type_str:
                     actual_type = type(value)
                     type_mismatch = False
                     corrected_value = value # Keep original unless corrected

                     if expected_type_str == "string" and actual_type is not str:
                         # Try to convert non-strings to string, flag for review
                         try:
                             corrected_value = str(value)
                             logger.warning(f"Call #{i+1} '{api_name}' argument '{param}': Auto-corrected type {actual_type.__name__} to string.")
                             needs_review = True
                         except Exception:
                             type_mismatch = True # Cannot convert
                     elif expected_type_str == "number":
                         if actual_type is not int and actual_type is not float:
                             # Try converting string numbers
                             if actual_type is str:
                                 try:
                                     corrected_value = float(value) # Convert to float first
                                     if corrected_value.is_integer():
                                         corrected_value = int(corrected_value) # Make int if possible
                                     logger.warning(f"Call #{i+1} '{api_name}' argument '{param}': Auto-corrected type string to {type(corrected_value).__name__}.")
                                     needs_review = True
                                 except ValueError:
                                     type_mismatch = True # Cannot convert string
                             else:
                                 type_mismatch = True # Not int, float, or convertible string
                     elif expected_type_str == "integer":
                         if actual_type is not int:
                             if actual_type is float and value.is_integer():
                                 corrected_value = int(value)
                                 logger.warning(f"Call #{i+1} '{api_name}' argument '{param}': Auto-corrected type float to int.")
                                 needs_review = True
                             elif actual_type is str:
                                 try:
                                     float_val = float(value)
                                     if float_val.is_integer():
                                         corrected_value = int(float_val)
                                         logger.warning(f"Call #{i+1} '{api_name}' argument '{param}': Auto-corrected type string to int.")
                                         needs_review = True
                                     else:
                                         type_mismatch = True # String represents a float, not int
                                 except ValueError:
                                     type_mismatch = True # Cannot convert string
                             else:
                                 type_mismatch = True # Not int or convertible float/string
                     elif expected_type_str == "boolean":
                         if actual_type is not bool:
                             # Try converting common string representations
                             if actual_type is str:
                                 val_lower = value.lower()
                                 if val_lower == "true":
                                     corrected_value = True
                                     logger.warning(f"Call #{i+1} '{api_name}' argument '{param}': Auto-corrected type string 'true' to boolean.")
                                     needs_review = True
                                 elif val_lower == "false":
                                     corrected_value = False
                                     logger.warning(f"Call #{i+1} '{api_name}' argument '{param}': Auto-corrected type string 'false' to boolean.")
                                     needs_review = True
                                 else:
                                     type_mismatch = True # Unrecognized string for boolean
                             else:
                                 type_mismatch = True # Not bool or convertible string

                     if type_mismatch:
                         return False, f"Call #{i+1} '{api_name}' argument '{param}': Expected type '{expected_type_str}', but got '{actual_type.__name__}'", data, False, {}
                     else:
                         # Update the arguments dict with the potentially corrected value
                         call["arguments"][param] = corrected_value


        # Plausibility Check
        if ENABLE_PLAUSIBILITY_CHECK:
            is_plausible, issues = check_argument_plausibility(api_name, call["arguments"], API_LIBRARY)
            if issues:
                plausibility_warnings.extend([f"Call #{i+1} '{api_name}': {issue}" for issue in issues])
                if not is_plausible:
                    # If check returns False, it's a hard fail for this stage
                    return False, f"Argument Plausibility Failed: {issues[0]}", data, True, {} # Mark for review on fail
                else:
                    # If check returns True but has issues, it's a warning
                    needs_review = True # Flag potential issues

        validated_calls.append(call)

    # If all checks pass for all calls
    validated_data = {"query": data["query"], "answer": validated_calls}
    reason = "Format OK"
    if plausibility_warnings:
        reason += f" (Plausibility Warnings: {'; '.join(plausibility_warnings)})"
        logger.warning(f"  Plausibility Warnings: {'; '.join(plausibility_warnings)}")

    return True, reason, validated_data, needs_review, generation_meta


def stage2_execution_checker(validated_data, generation_meta):
    """Attempts to 'execute' the validated function calls using simulators."""
    query = validated_data.get("query", "")
    calls_to_execute = validated_data.get("answer", [])
    execution_results = []
    all_calls_succeeded = True

    if not calls_to_execute:
        logger.info("  No API calls to execute.")
        # Still return success if no calls were intended (e.g., negative sample)
        # Semantic check will handle if calls *should* have been present

    for i, call in enumerate(calls_to_execute):
        api_name = call.get("name")
        arguments = call.get("arguments", {})
        result = {"call": call} # Store the original call

        if api_name in API_EXECUTORS:
            executor_func = API_EXECUTORS[api_name]
            try:
                # Simulate execution
                output = executor_func(**arguments)
                result["execution_success"] = True
                result["execution_output"] = output
                logger.info(f"    Call #{i+1} '{api_name}': Execution simulation SUCCEEDED.")
                logger.debug(f"      Output: {output}")
            except Exception as e:
                result["execution_success"] = False
                result["execution_output"] = {"error": str(e)}
                all_calls_succeeded = False
                logger.warning(f"    Call #{i+1} '{api_name}': Execution simulation FAILED: {e}")
        else:
            result["execution_success"] = False
            result["execution_output"] = {"error": f"No simulator found for API '{api_name}'"}
            all_calls_succeeded = False
            logger.error(f"    Call #{i+1} '{api_name}': No execution simulator found.")

        execution_results.append(result)

    # Pass generation_meta through
    passing_data = {
        "query": query,
        "execution_results": execution_results,
        **generation_meta # Add metadata back
    }
    # Log overall execution outcome for the sample
    if calls_to_execute: # Only log if there were calls to attempt
        if all_calls_succeeded:
            logger.info("  All execution simulations PASSED.")
        else:
            logger.warning("  One or more execution simulations FAILED.")

    return all_calls_succeeded, "Execution attempted", passing_data


def stage3_semantic_checker(execution_data):
    """Performs semantic check based on keywords, context, and negative sampling target."""
    query = execution_data.get("query", "").lower()
    results = execution_data.get("execution_results", [])
    is_negative_target = execution_data.get("is_negative_sample_target", False)
    needs_review = execution_data.get("needs_review", False) # Carry over flag

    executed_successfully = [res for res in results if res.get("execution_success")]
    num_successful_calls = len(executed_successfully)

    # Handle Negative Sampling Target
    if is_negative_target:
        if num_successful_calls == 0:
            logger.info("  Semantic PASSED (Targeted negative sample with no successful calls)")
            # Add negative sample flag to final data
            execution_data["is_negative_sample"] = True
            return True, "Passed (Negative sample)", execution_data, needs_review
        else:
            logger.warning("  Semantic FAILED (Targeted negative sample but had successful calls)")
            needs_review = True # Definitely needs review
            return False, "Failed (Negative sample target had successful calls)", execution_data, needs_review

    # Handle cases where no API calls were made/succeeded (and not targeted negative)
    if num_successful_calls == 0:
        # Check if query looks like a simple conversational turn
        # ... (existing conversational check) ...
        if any(phrase in query for phrase in conversational_phrases):
             logger.info("  Semantic PASSED (Conversational query, no successful API calls needed)")
             return True, "Passed (Conversational query)", execution_data, needs_review
        # Check if it's a simple question
        # ... (existing question check) ...
        if any(word in query for word in question_words):
             logger.info("  Semantic PASSED (General question, no successful API calls needed)")
             return True, "Passed (General question)", execution_data, needs_review

        # If not conversational/question and no calls succeeded, it's likely a failure
        logger.warning("  Semantic FAILED (Query likely needs API calls but none succeeded)")
        needs_review = True
        return False, "Failed (No successful calls for non-trivial query)", execution_data, needs_review

    # Keyword Check (for positive samples)
    expected_apis = set()
    # ... (existing keyword detection) ...

    if not expected_apis:
        logger.info("  Semantic PASSED (No specific keywords detected, but calls succeeded)")
        # Could add needs_review=True here if we want to be cautious
        return True, "Passed (No keywords, calls succeeded)", execution_data, needs_review

    # Check overlap between expected and successfully executed APIs
    executed_api_names = {res["call"]["name"] for res in executed_successfully}

    if expected_apis.intersection(executed_api_names):
        logger.info(f"  Semantic PASSED (Keyword match: Expected {expected_apis}, Got {executed_api_names})")
        # Check if ALL expected APIs were called? Maybe add needs_review if some are missing.
        if not expected_apis.issubset(executed_api_names):
             logger.warning(f"    Missing expected APIs: {expected_apis - executed_api_names}")
             needs_review = True
        return True, "Passed (Keyword match)", execution_data, needs_review
    else:
        # Embedding Similarity Check (Placeholder)
        if ENABLE_SEMANTIC_SIMILARITY_CHECK:
             scores = []
             for res in executed_successfully:
                 api_name = res["call"]["name"]
                 api_desc = API_LIBRARY.get(api_name, {}).get("description", "")
                 if api_desc:
                     score = check_semantic_similarity(query, api_desc, SEMANTIC_EMBEDDING_MODEL)
                     scores.append(score)
             avg_score = sum(scores) / len(scores) if scores else 0
             similarity_threshold = 0.5 # Example threshold
             if avg_score >= similarity_threshold:
                 logger.info(f"  Semantic PASSED (Embedding similarity {avg_score:.2f} >= {similarity_threshold})")
                 needs_review = True # Flag similarity passes for review
                 return True, f"Passed (Similarity {avg_score:.2f})", execution_data, needs_review
             else:
                 logger.warning(f"  Semantic FAILED (Keyword mismatch AND low similarity {avg_score:.2f})")

        # If no keyword match and similarity check is off or failed
        expected_str = ', '.join(expected_apis)
        executed_str = ', '.join(executed_api_names) if executed_api_names else 'None'
        fail_reason = f"Failed: Query keywords suggest [{expected_str}], " \
                      f"but successful executions were [{executed_str}]"
        logger.warning(f"  {fail_reason}")
        needs_review = True
        return False, fail_reason, execution_data, needs_review


# --- Main Pipeline Orchestration ---

def process_existing_data(file_path, simplified_only=False, visualize_only=False):
    """Process existing verified data without generating new examples."""
    # Load existing data
    with open(file_path, 'r', encoding='utf-8') as f:
        existing_data = [json.loads(line) for line in f]

    if simplified_only:
        simplified_data = convert_to_simplified_format(existing_data)
        simplified_path = os.path.join(output_dir, SIMPLIFIED_DATA_FILENAME)
        if save_jsonl(simplified_data, simplified_path):
            logger.info(f"Saved simplified format to: {simplified_path}")
        else:
            logger.error(f"Error saving simplified data to {simplified_path}")

    if visualize_only:
        stats_data = generate_statistics(existing_data)
        stats_path = os.path.join(output_dir, STATISTICS_FILENAME)
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved statistics to {stats_path}")
            visualize_statistics(stats_data, VISUALIZATIONS_DIR)
        except Exception as e:
            logger.error(f"Error generating/saving statistics or visualizations: {e}")

def run_pipeline(args):
    """Runs the full APIGen pipeline with LLM generation and validation."""
    # Initialize statistics
    stats = {
        "total_attempted": 0,
        "fail_format": 0,
        "fail_execution": 0,
        "fail_semantic": 0,
        "pass": 0
    }

    # Load existing data if available
    existing_data = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = [json.loads(line) for line in f]

    verified_data_all_stages = []

    # Main loop for generating and validating data
    for i in range(args.num_samples):
        start_time = time.time()
        stats["total_attempted"] += 1

        # Stage 1: Format Check
        format_ok, reason1, validated_data = stage1_format_checker()
        logger.info(f"Stage 1 (Format Check): {'PASS' if format_ok else 'FAIL'} - {reason1}")
        if not format_ok:
            stats["fail_format"] += 1
            continue

        # Stage 2: Execution Check
        if ENABLE_EXECUTION_CHECK:
            exec_ok, reason2, execution_data = stage2_execution_checker(validated_data)
            logger.info(f"Stage 2 (Execution Check): {'ALL PASS' if exec_ok else 'SOME FAIL/NO CALLS'} - {reason2}")
            if not exec_ok and validated_data.get("answer"):
                stats["fail_execution"] += 1
                continue
        else:
            execution_data = {"query": validated_data["query"], "execution_results": []}
            for call in validated_data.get("answer", []):
                execution_data["execution_results"].append({
                    "call": call,
                    "execution_success": True,
                    "execution_output": {"simulated": "Execution check disabled"}
                })
            exec_ok = True
            logger.info("Stage 2 (Execution Check): SKIPPED (disabled)")

        # Stage 3: Semantic Check
        if ENABLE_SEMANTIC_CHECK:
            semantic_ok, reason3, final_data = stage3_semantic_checker(execution_data)
            logger.info(f"Stage 3 (Semantic Check): {'PASS' if semantic_ok else 'FAIL'} - {reason3}")
            if not semantic_ok:
                stats["fail_semantic"] += 1
                continue
        else:
            final_data = execution_data
            semantic_ok = True
            logger.info("Stage 3 (Semantic Check): SKIPPED (disabled)")

        if format_ok and semantic_ok:
            stats["pass"] += 1
            verified_data_all_stages.append(final_data)
            logger.info(f"--- Output #{i+1} PASSED all checks ({time.time() - start_time:.2f}s) ---")
        else:
            logger.warning(f"--- Output #{i+1} FAILED one or more checks ({time.time() - start_time:.2f}s) ---")

    logger.info("\n--- Pipeline Summary ---")
    logger.info(f"Total LLM outputs attempted: {stats['total_attempted']}")
    logger.info(f"Failed Format Check: {stats['fail_format']}")
    logger.info(f"Failed Execution Check (leading to skip): {stats['fail_execution']}")
    logger.info(f"Failed Semantic Check: {stats['fail_semantic']}")
    logger.info(f"Passed All Checks: {stats['pass']}")

    final_verified_data = existing_data + verified_data_all_stages
    logger.info(f"Total verified examples (existing + new): {len(final_verified_data)}")

    if final_verified_data:
        if save_jsonl(final_verified_data, output_path):
            logger.info(f"Saved {len(final_verified_data)} verified data points to: {output_path}")
        else:
            logger.error(f"Error saving verified data to {output_path}")

        if SAVE_CSV_OUTPUT:
            csv_path = os.path.join(output_dir, VERIFIED_DATA_CSV_FILENAME)
            csv_data = []
            for item in final_verified_data:
                query = item.get("query", "")
                calls_str = json.dumps([res["call"] for res in item.get("execution_results", []) if res.get("call")], ensure_ascii=False)
                results_str = json.dumps([{"success": res["execution_success"], "output": res["execution_output"]} for res in item.get("execution_results", [])], ensure_ascii=False)
                csv_data.append({"query": query, "api_calls": calls_str, "execution_results": results_str})

            if save_csv(csv_data, csv_path, columns=["query", "api_calls", "execution_results"]):
                logger.info(f"Saved CSV format to: {csv_path}")
            else:
                logger.error(f"Error saving CSV data to {csv_path}")

        if SAVE_SIMPLIFIED_FORMAT:
            simplified_path = os.path.join(output_dir, SIMPLIFIED_DATA_FILENAME)
            simplified_data = convert_to_simplified_format(final_verified_data)
            if save_jsonl(simplified_data, simplified_path):
                logger.info(f"Saved simplified format to: {simplified_path}")
            else:
                logger.error(f"Error saving simplified data to {simplified_path}")

        if GENERATE_VISUALIZATIONS:
            stats_data = generate_statistics(final_verified_data)
            stats_path = os.path.join(output_dir, STATISTICS_FILENAME)
            try:
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved statistics to {stats_path}")
                visualize_statistics(stats_data, VISUALIZATIONS_DIR)
            except Exception as e:
                logger.error(f"Error generating/saving statistics or visualizations: {e}")
    else:
        logger.warning("\nNo data passed all verification stages. No output files generated/updated.")

if __name__ == "__main__":
    args = parse_arguments()

    if response_cache and not args.use_cache:
        logger.info("Disabling LLM response cache based on command-line argument.")
        response_cache = None

    run_pipeline(args)

    logger.info("\nAPIGen LLM script finished.")