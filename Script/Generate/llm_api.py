import os
import requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import logging
import json
from available_tools import AVAILABLE_TOOLS, TOOL_FUNCTIONS
from super_gen_data import SuperGenData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

OLLAMA_API_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = "mistral"  # Change this to match your Ollama model name

# Serve OpenAPI spec
@app.route('/openapi.yaml')
def serve_openapi_spec():
    return send_from_directory(os.path.dirname(__file__), 'openapi.yaml')

# Serve Swagger HTML
@app.route('/')
@app.route('/docs')
def serve_swagger():
    with open(os.path.join(os.path.dirname(__file__), 'swagger.html'), 'r') as f:
        content = f.read()
    return content

# Generate text endpoint
@app.route('/generate', methods=['POST'])
def generate_text():
    # Handle both JSON and form-data requests
    if request.is_json:
        data = request.get_json()
        prompt = data.get('prompt', '')
        tools = data.get('tools', None)
        image = None
    else:
        prompt = request.form.get('prompt', '')
        tools = request.form.get('tools', None)
        image = request.files.get('image', None)
    
    if not prompt:
        return jsonify({'error': 'Missing prompt in request'}), 400
    
    # Handle file upload for OCR
    if image:
        image_path = os.path.join(os.path.dirname(__file__), 'temp_image.jpg')
        image.save(image_path)
        ocr_result = TOOL_FUNCTIONS['perform_ocr'](image_path)
        ocr_data = json.loads(ocr_result)
        if 'error' in ocr_data:
            return jsonify({'error': ocr_data['error']}), 500
        prompt = f"ข้อความที่ได้จาก OCR: {ocr_data['text']}\n\n{prompt}"
    
    # Format the prompt for raw mode if tools are provided
    if tools:
        formatted_prompt = f"[AVAILABLE_TOOLS]{json.dumps(tools)}[/AVAILABLE_TOOLS][INST]{prompt}[/INST]"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": formatted_prompt,
            "stream": True,
            "raw": True
        }
        return handle_function_calling(payload)
    else:
        # Format prompt for normal text generation
        formatted_prompt = f"""You are an expert assistant with advanced knowledge. Provide detailed and comprehensive responses in Thai:
{prompt}"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": formatted_prompt,
            "stream": True
        }
        return handle_text_generation(payload)

def handle_function_calling(payload):
    def generate():
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        chunk = json.loads(decoded_line)
                        if 'response' in chunk:
                            yield f"data: {chunk['response']}\n\n"
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON: {decoded_line}")
                        yield f"data: [ERROR] Failed to parse response chunk\n\n"
            
            yield "data: [DONE]\n\n"
            
        except requests.RequestException as e:
            logger.error(f"Failed to generate text: {str(e)}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def handle_text_generation(payload):
    def generate():
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        chunk = json.loads(decoded_line)
                        if 'response' in chunk:
                            yield f"data: {chunk['response']}\n\n"
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON: {decoded_line}")
                        yield f"data: [ERROR] Failed to parse response chunk\n\n"
            
            yield "data: [DONE]\n\n"
            
        except requests.RequestException as e:
            logger.error(f"Failed to generate text: {str(e)}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    logger.info("Starting LLM API server...")
    logger.info(f"API Documentation available at http://localhost:5000/docs")
    logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
    
    # Initialize SuperGenData class with the app instance
    super_gen = SuperGenData(app, OLLAMA_API_URL, OLLAMA_MODEL)
    
    app.run(debug=True)