#!/usr/bin/env python3
"""
Startup script for Stock Analysis System
Starts the API server and provides instructions for frontend
"""

import sys
import os
import subprocess
import time
import webbrowser
from threading import Thread

def check_dependencies():
    """Check if all dependencies are installed"""
    print("Checking dependencies...")
    
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'yfinance', 'prophet',
        'sklearn', 'fastapi', 'uvicorn', 'vaderSentiment', 'textblob', 'xgboost'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except ImportError:
            print(f"[MISSING] {module}")
            missing.append(module)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("All dependencies are installed!")
    return True

def start_api_server():
    """Start the API server"""
    print("\nStarting API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation going at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the API server using uvicorn with the correct module path
        # The API is located in data_ingestion/api/main.py
        print("Starting uvicorn server...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "data_ingestion.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\nAPI server stopped by user")
    except Exception as e:
        print(f"Error starting API server: {e}")

def open_browser():
    """Open browser to API docs after a delay"""
    time.sleep(3)
    try:
        webbrowser.open("http://localhost:8000/docs")
    except:
        pass

def main():
    """Main startup function"""
    print("=" * 60)
    print("Stock Analysis System - Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies first:")
        print("pip install -r requirements.txt")
        return
    
    print("\n" + "=" * 60)
    print("SYSTEM READY!")
    print("=" * 60)
    
    print("\nAvailable components:")
    print("1. API Server (FastAPI)")
    print("2. React Frontend Dashboard")
    print("3. Prophet Time Series Model")
    print("4. XGBoost Machine Learning Model")
    print("5. Sentiment Analysis")
    print("6. Comprehensive Evaluation")
    print("7. Interactive Visualizations")
    
    print("\n" + "=" * 60)
    print("STARTING API SERVER")
    print("=" * 60)
    
    # Open browser in background
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start API server
    start_api_server()

if __name__ == "__main__":
    main()

