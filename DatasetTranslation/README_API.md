# Dataset Translation API

RESTful API service for translating datasets using DeepL API with validation and background job processing.

## Setup

1. Install dependencies:
```bash
pip install -r api_requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your DeepL API key
```

3. Start the API server:
```bash
uvicorn api:app --reload
```

## API Endpoints

### Languages
- `GET /languages`
  - Get list of supported languages
  - Returns: `{"languages": ["EN", "DE", ...]}`

### Single Text Translation
- `POST /translate/text`
  ```json
  {
    "text": "Hello world",
    "source_lang": "EN",
    "target_lang": "TH"
  }
  ```
  - Returns translated text with validation results

### Batch Translation
- `POST /translate/batch`
  ```json
  {
    "texts": ["Text 1", "Text 2"],
    "source_lang": "EN",
    "target_lang": "TH",
    "batch_size": 10
  }
  ```
  - Returns job ID for tracking progress
  - Processes translations in background

### Translation Validation
- `POST /validate`
  ```json
  {
    "original": "Hello",
    "translated": "สวัสดี",
    "source_lang": "EN",
    "target_lang": "TH"
  }
  ```
  - Returns validation results and detected languages

### Dataset Translation
- `POST /dataset`
  - Form data with file upload
  - Parameters:
    - `file`: Dataset file (CSV/JSON/JSONL)
    - `source_lang`: Source language
    - `target_lang`: Target language
    - `batch_size`: Batch size (default: 10)
  - Returns job ID for tracking

### Job Status
- `GET /jobs/{job_id}`
  - Get translation job status and results
  - Returns:
    ```json
    {
      "job_id": "job_1",
      "total_items": 100,
      "completed_items": 50,
      "status": "in_progress",
      "results": [...],
      "errors": null
    }
    ```

## Features

- Asynchronous processing with background tasks
- Batch processing with configurable batch size
- Translation validation and quality checks
- Language detection
- Progress tracking
- Comprehensive error handling
- Support for multiple file formats

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

Error responses include detailed messages:
```json
{
  "detail": "Error message here"
}
```

## Example Usage

### Using curl

```bash
# Translate single text
curl -X POST "http://localhost:8000/translate/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "source_lang": "EN", "target_lang": "TH"}'

# Upload dataset
curl -X POST "http://localhost:8000/dataset" \
     -F "file=@dataset.csv" \
     -F "source_lang=EN" \
     -F "target_lang=TH"

# Check job status
curl "http://localhost:8000/jobs/job_1"
```

### Using Python

```python
import requests

# Translate single text
response = requests.post(
    "http://localhost:8000/translate/text",
    json={
        "text": "Hello world",
        "source_lang": "EN",
        "target_lang": "TH"
    }
)
print(response.json())

# Upload dataset
with open('dataset.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/dataset",
        files={"file": f},
        data={
            "source_lang": "EN",
            "target_lang": "TH"
        }
    )
job_id = response.json()["job_id"]

# Check job status
status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
print(status)
```

## Performance Considerations

- Uses asyncio for concurrent processing
- Implements batch processing to optimize API usage
- Background task processing for large datasets
- Memory-efficient streaming for large files