"""
Ticker loading functionality for U.S. stocks (NASDAQ + NYSE)
Includes fallback mechanisms to ensure non-empty ticker list
"""

import os
import json
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)

# Fallback ticker list - most common U.S. stocks (NASDAQ + NYSE)
FALLBACK_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "NFLX", "CRM",
    "ORCL", "INTC", "AMD", "QCOM", "AVGO", "TXN", "ADBE", "CSCO", "NOW", "INTU",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
    # Consumer
    "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "DIS", "CMCSA", "NFLX",
    # Healthcare
    "JNJ", "PFE", "UNH", "ABT", "TMO", "ABBV", "MRK", "BMY", "AMGN", "GILD",
    # Industrial
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "EMR", "ETN",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "HAL", "OXY",
    # Consumer Staples
    "PEP", "KO", "PG", "CL", "MO", "PM", "STZ", "TAP", "BF.B", "CHD",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "SRE", "EXC", "XEL", "ES", "PEG", "ED",
    # Real Estate
    "AMT", "PLD", "EQIX", "PSA", "WELL", "SPG", "O", "DLR", "AVB", "EQR",
    # Materials
    "LIN", "APD", "ECL", "SHW", "DD", "PPG", "FCX", "NEM", "VMC", "MLM",
    # Communication
    "T", "VZ", "TMUS", "LUMN", "ATUS", "CABO", "SHEN", "USM", "CNSL", "VSAT"
]

def get_fallback_tickers() -> List[str]:
    """Return fallback ticker list if all other methods fail"""
    return FALLBACK_TICKERS.copy()

def load_tickers_from_file(file_path: str = None) -> List[str]:
    """
    Load tickers from a local JSON or CSV file
    Returns empty list if file doesn't exist or can't be read
    """
    if file_path is None:
        # Try common locations (relative to project root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        possible_paths = [
            os.path.join(project_root, "data", "tickers.json"),
            os.path.join(script_dir, "..", "data", "tickers.json"),
            "data/tickers.json",
            "tickers.json",
            "data/us_tickers.json",
            "stockproject/data/tickers.json"
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                file_path = abs_path
                break
        
        if file_path is None:
            return []
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(t).upper().strip() for t in data if t]
                elif isinstance(data, dict) and 'tickers' in data:
                    return [str(t).upper().strip() for t in data['tickers'] if t]
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Try common column names
            for col in ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER', 'SYMBOL']:
                if col in df.columns:
                    return [str(t).upper().strip() for t in df[col].dropna().tolist() if t]
            # If no matching column, use first column
            if len(df.columns) > 0:
                return [str(t).upper().strip() for t in df.iloc[:, 0].dropna().tolist() if t]
    except Exception as e:
        logger.warning(f"Failed to load tickers from file {file_path}: {e}")
    
    return []

def load_tickers_from_yfinance() -> List[str]:
    """
    Attempt to load tickers using yfinance
    This is a fallback method that may not always work
    """
    try:
        import yfinance as yf
        
        # Try to get NASDAQ and NYSE tickers
        # Note: yfinance doesn't have a direct method, so we'll use fallback
        # In a real scenario, you might use nasdaqdatalink or other APIs
        return []
    except Exception as e:
        logger.warning(f"Failed to load tickers from yfinance: {e}")
        return []

def load_tickers() -> List[str]:
    """
    Main function to load U.S. stock tickers
    Tries multiple methods with fallback to ensure non-empty list
    """
    tickers = []
    
    # Method 1: Try loading from file
    tickers = load_tickers_from_file()
    if tickers:
        logger.info(f"Loaded {len(tickers)} tickers from file")
        return tickers
    
    # Method 2: Try yfinance (if available)
    tickers = load_tickers_from_yfinance()
    if tickers:
        logger.info(f"Loaded {len(tickers)} tickers from yfinance")
        return tickers
    
    # Method 3: Use fallback list (always works)
    logger.info(f"Using fallback ticker list with {len(FALLBACK_TICKERS)} tickers")
    return get_fallback_tickers()

def save_tickers_to_file(tickers: List[str], file_path: str = "data/tickers.json"):
    """
    Save ticker list to a JSON file for future use
    """
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump({"tickers": tickers}, f, indent=2)
        logger.info(f"Saved {len(tickers)} tickers to {file_path}")
    except Exception as e:
        logger.warning(f"Failed to save tickers to file: {e}")

if __name__ == "__main__":
    # Test the ticker loader
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers")
    print(f"First 20: {tickers[:20]}")

