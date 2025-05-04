"""
Main entry point for running both API and Streamlit interfaces.
"""
import argparse
import os
import uvicorn
import subprocess
from config import API_HOST, API_PORT, API_WORKERS

def run_api():
    """Run FastAPI server."""
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=True
    )

def run_streamlit():
    """Run Streamlit interface."""
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

def run_both():
    """Run both API and Streamlit in parallel."""
    api_process = subprocess.Popen(["python", "-m", "uvicorn", "api:app", 
                                  "--host", API_HOST, "--port", str(API_PORT),
                                  "--workers", str(API_WORKERS), "--reload"])
    
    streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
    
    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        api_process.terminate()
        streamlit_process.terminate()
        api_process.wait()
        streamlit_process.wait()

def main():
    parser = argparse.ArgumentParser(description="Run Dataset Translation services")
    parser.add_argument(
        "--mode",
        choices=["api", "streamlit", "both"],
        default="both",
        help="Which service to run (default: both)"
    )

    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "streamlit":
        run_streamlit()
    else:
        run_both()

if __name__ == "__main__":
    main()