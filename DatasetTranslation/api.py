from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import asyncio
import logging
from config import (
    API_HOST, API_PORT, API_WORKERS, OUTPUT_DIR,
    SUPPORTED_FILE_TYPES, MAX_FILE_SIZE, BATCH_SIZE
)
from shared_utils import TranslationManager

# Initialize FastAPI app
app = FastAPI(
    title="Dataset Translation API",
    description="API for translating datasets using DeepL",
    version="1.0.0"
)

# Initialize translation manager
try:
    translation_manager = TranslationManager()
except ValueError as e:
    logging.error(f"Failed to initialize TranslationManager: {e}")
    raise

# Models
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_lang: str
    target_lang: str
    batch_size: Optional[int] = BATCH_SIZE

class ValidationRequest(BaseModel):
    original: str
    translated: str
    source_lang: str
    target_lang: str

class TranslationJob(BaseModel):
    job_id: str
    total_items: int
    completed_items: int
    status: str
    results: Optional[List[Dict]] = None
    errors: Optional[List[str]] = None

# In-memory job storage (replace with database in production)
translation_jobs = {}

@app.get("/languages")
async def get_languages():
    """Get list of supported languages."""
    return {"languages": translation_manager.get_supported_languages()}

@app.post("/translate/text")
async def translate_single(request: TranslationRequest):
    """Translate a single text."""
    try:
        result = await translation_manager.translate_text(
            request.text,
            request.source_lang,
            request.target_lang
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/batch")
async def translate_batch(request: BatchTranslationRequest):
    """Start a batch translation job."""
    try:
        job_id = f"job_{len(translation_jobs)}"
        translation_jobs[job_id] = TranslationJob(
            job_id=job_id,
            total_items=len(request.texts),
            completed_items=0,
            status="in_progress"
        )

        # Start background task
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            process_batch_translation,
            job_id,
            request.texts,
            request.source_lang,
            request.target_lang,
            request.batch_size
        )

        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_translation(
    job_id: str,
    texts: List[str],
    source_lang: str,
    target_lang: str,
    batch_size: int
):
    """Process batch translation in background."""
    job = translation_jobs[job_id]
    
    try:
        # Use translation manager for batch translation
        def progress_callback(progress: float):
            job.completed_items = int(progress * len(texts))

        results = await translation_manager.translate_batch(
            texts, source_lang, target_lang,
            batch_size, progress_callback
        )
        
        job.status = "completed"
        job.results = results
    except Exception as e:
        job.status = "failed"
        job.errors = [str(e)]
        logging.error(f"Batch translation failed: {e}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a translation job."""
    if job_id not in translation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return translation_jobs[job_id]

@app.post("/validate")
async def validate_translation_pair(request: ValidationRequest):
    """Validate a translation pair."""
    is_valid, issues = await translation_manager.validate_translation(
        request.original,
        request.translated,
        request.source_lang,
        request.target_lang
    )
    return {
        "is_valid": is_valid,
        "issues": issues,
        "detected_source_lang": translation_manager.detect_language(request.original),
        "detected_target_lang": translation_manager.detect_language(request.translated)
    }

@app.post("/dataset")
async def translate_dataset(
    file: UploadFile = File(...),
    source_lang: str = None,
    target_lang: str = None,
    batch_size: int = BATCH_SIZE
):
    """Translate an entire dataset file."""
    # Validate file
    if file.content_type not in SUPPORTED_FILE_TYPES.values():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    try:
        # Load dataset
        dataset = translation_manager.load_dataset(file.file)
        
        # Start translation job
        job_id = f"dataset_job_{len(translation_jobs)}"
        translation_jobs[job_id] = TranslationJob(
            job_id=job_id,
            total_items=len(dataset),
            completed_items=0,
            status="in_progress"
        )

        # Start background task
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            process_dataset_translation,
            job_id,
            dataset,
            source_lang,
            target_lang,
            batch_size
        )

        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_dataset_translation(
    job_id: str,
    dataset: List[Dict],
    source_lang: str,
    target_lang: str,
    batch_size: int
):
    """Process dataset translation in background."""
    job = translation_jobs[job_id]
    
    try:
        translated_dataset = []
        total_items = len(dataset)
        
        for i in range(0, total_items, batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = []

            for item in batch:
                # Translate each text field in the item
                translated_item = {}
                for key, value in item.items():
                    if isinstance(value, str):
                        result = await translation_manager.translate_text(
                            value, source_lang, target_lang
                        )
                        translated_item[f"{key}_translated"] = result
                    else:
                        translated_item[key] = value
                batch_results.append(translated_item)

            translated_dataset.extend(batch_results)
            job.completed_items = i + len(batch)

        job.status = "completed"
        job.results = translated_dataset
    except Exception as e:
        job.status = "failed"
        job.errors = [str(e)]
        logging.error(f"Dataset translation failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)