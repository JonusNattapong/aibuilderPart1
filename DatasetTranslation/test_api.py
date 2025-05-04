import pytest
from fastapi.testclient import TestClient
from api import app
import json
import os
from pathlib import Path

client = TestClient(app)

# Test data
TEST_TEXT = "Hello world"
TEST_BATCH = ["Hello world", "How are you?"]
TEST_DATASET = [
    {"text": "Hello world", "id": 1},
    {"text": "How are you?", "id": 2}
]

def test_get_languages():
    """Test getting supported languages."""
    response = client.get("/languages")
    assert response.status_code == 200
    assert "languages" in response.json()
    assert len(response.json()["languages"]) > 0

def test_translate_text():
    """Test single text translation."""
    request_data = {
        "text": TEST_TEXT,
        "source_lang": "EN",
        "target_lang": "TH"
    }
    response = client.post("/translate/text", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "original" in data
    assert "translated" in data
    assert "detected_lang" in data
    assert "needs_review" in data
    assert "issues" in data

def test_translate_batch():
    """Test batch translation."""
    request_data = {
        "texts": TEST_BATCH,
        "source_lang": "EN",
        "target_lang": "TH",
        "batch_size": 2
    }
    response = client.post("/translate/batch", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data

def test_validate_translation():
    """Test translation validation."""
    request_data = {
        "original": "Hello",
        "translated": "สวัสดี",
        "source_lang": "EN",
        "target_lang": "TH"
    }
    response = client.post("/validate", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "is_valid" in data
    assert "issues" in data
    assert "detected_source_lang" in data
    assert "detected_target_lang" in data

def test_translate_dataset():
    """Test dataset translation."""
    # Create test dataset file
    test_file = Path("test_dataset.json")
    test_file.write_text(json.dumps(TEST_DATASET))

    with open(test_file, "rb") as f:
        response = client.post(
            "/dataset",
            files={"file": f},
            data={
                "source_lang": "EN",
                "target_lang": "TH",
                "batch_size": 2
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data

    # Cleanup
    test_file.unlink()

def test_get_job_status():
    """Test getting job status."""
    # First create a job
    request_data = {
        "texts": TEST_BATCH,
        "source_lang": "EN",
        "target_lang": "TH"
    }
    response = client.post("/translate/batch", json=request_data)
    job_id = response.json()["job_id"]

    # Then check its status
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "total_items" in data
    assert "completed_items" in data
    assert "status" in data

def test_invalid_language():
    """Test error handling for invalid language."""
    request_data = {
        "text": TEST_TEXT,
        "source_lang": "XX",  # Invalid language code
        "target_lang": "TH"
    }
    response = client.post("/translate/text", json=request_data)
    assert response.status_code == 400 or response.status_code == 500

def test_missing_api_key():
    """Test error handling when API key is missing."""
    # Temporarily remove API key from environment
    api_key = os.environ.pop("DEEPL_API_KEY", None)
    
    request_data = {
        "text": TEST_TEXT,
        "source_lang": "EN",
        "target_lang": "TH"
    }
    response = client.post("/translate/text", json=request_data)
    assert response.status_code == 500

    # Restore API key
    if api_key:
        os.environ["DEEPL_API_KEY"] = api_key

def test_invalid_job_id():
    """Test error handling for invalid job ID."""
    response = client.get("/jobs/invalid_job_id")
    assert response.status_code == 404

def test_invalid_file_format():
    """Test error handling for invalid file format."""
    # Create test file with invalid format
    test_file = Path("test_invalid.txt")
    test_file.write_text("Invalid format")

    with open(test_file, "rb") as f:
        response = client.post(
            "/dataset",
            files={"file": f},
            data={
                "source_lang": "EN",
                "target_lang": "TH"
            }
        )
    
    assert response.status_code == 400 or response.status_code == 500

    # Cleanup
    test_file.unlink()