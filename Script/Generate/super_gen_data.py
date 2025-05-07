import json
import logging
from flask import jsonify, request
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SuperGenData:
    def __init__(self, app, model_url: str = 'http://localhost:11434/api/generate', model_name: str = 'mistral'):
        self.app = app
        self.OLLAMA_API_URL = model_url
        self.OLLAMA_MODEL = model_name
        self.dataset_types = {
            'nlp': 'NLP',
            'vision': 'Computer Vision',
            'reasoning': 'Reasoning'
        }
        self.register_routes()

    def _format_prompt(self, dataset_type: str, specific_type: str, prompt: str) -> str:
        """Format prompt based on dataset type"""
        dataset_name = self.dataset_types.get(dataset_type, dataset_type.upper())
        return f"""คุณเป็น AI ผู้เชี่ยวชาญในการสร้างชุดข้อมูล {dataset_name}
โปรดสร้างชุดข้อมูล {specific_type} จากคำสั่งต่อไปนี้:
{prompt}"""

    def _create_payload(self, formatted_prompt: str, stream: bool = True) -> Dict[str, Any]:
        """Create payload for Ollama API request"""
        return {
            "model": self.OLLAMA_MODEL,
            "prompt": formatted_prompt,
            "stream": stream
        }

    def _handle_dataset_request(self, dataset_type: str) -> Any:
        """Handle individual dataset generation request"""
        try:
            data = request.get_json()
            specific_type = data.get('dataset_type', '')
            prompt = data.get('prompt', '')
            
            if not specific_type or not prompt:
                return jsonify({'error': 'Missing dataset_type or prompt in request'}), 400
            
            formatted_prompt = self._format_prompt(dataset_type, specific_type, prompt)
            payload = self._create_payload(formatted_prompt)
            
            from llm_api import handle_text_generation
            return handle_text_generation(payload)
            
        except Exception as e:
            logger.error(f"Failed to generate {dataset_type} dataset: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def register_routes(self):
        """Register all routes for dataset generation"""
        @self.app.route('/generate_nlp_dataset', methods=['POST'])
        def generate_nlp_dataset():
            return self._handle_dataset_request('nlp')

        @self.app.route('/generate_vision_dataset', methods=['POST'])
        def generate_vision_dataset():
            return self._handle_dataset_request('vision')

        @self.app.route('/generate_reasoning_dataset', methods=['POST'])
        def generate_reasoning_dataset():
            return self._handle_dataset_request('reasoning')