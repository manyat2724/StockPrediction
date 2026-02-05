"""
Anomaly Detection Router for Stock Risk Management
Provides statistical anomaly detection based on z-score of daily returns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Import data loading functions
from modeling.prophet_model import load_features
from data_ingestion.stock_fetch import fetch_stock_data
from feature_engineering.feature import simulate_sentiment_data, add_rolling_features

# Create router
router = APIRouter(prefix="/anomaly-detection", tags=["Anomaly Detection"])


class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    ticker: str
    use_real_sentiment: Optional[bool] = True


class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection"""
    ticker: str
    anomalies_detected: int
    risk_level: str  # LOW, MEDIUM, HIGH
    message: str
    details: Optional[dict] = None


def detect_anomalies_zscore(ticker: str, use_real_sentiment: bool = True) -> dict:
    """
    Statistical anomaly detection based on z-score of daily returns
    
    Args:
        ticker: Stock ticker symbol
        use_real_sentiment: Whether to use real sentiment data
        
    Returns:
        Dictionary with anomaly_count, risk_level, and message
    """
    try:
        # Load stock data
        df = load_features(ticker=ticker, use_real_sentiment=use_real_sentiment)
        
        if len(df) < 20:
            return {
                "anomalies_detected": 0,
                "risk_level": "LOW",
                "message": "Insufficient data for anomaly detection (minimum 20 data points required)",
                "details": {
                    "data_points": len(df),
                    "minimum_required": 20
                }
            }
        
        # Get price data
        if 'close_price' in df.columns:
            prices = df['close_price'].values
        elif 'Close' in df.columns:
            prices = df['Close'].values
        else:
            prices = df['y'].values if 'y' in df.columns else None
        
        if prices is None or len(prices) < 2:
            return {
                "anomalies_detected": 0,
                "risk_level": "LOW",
                "message": "No price data available for anomaly detection",
                "details": None
            }
        
        # Calculate daily returns
        returns = np.diff(prices) / prices[:-1]  # Percentage returns
        returns_pct = returns * 100  # Convert to percentage
        
        if len(returns) < 10:
            return {
                "anomalies_detected": 0,
                "risk_level": "LOW",
                "message": "Insufficient return data for statistical analysis",
                "details": {
                    "return_data_points": len(returns),
                    "minimum_required": 10
                }
            }
        
        # Calculate z-scores for returns
        mean_return = np.mean(returns_pct)
        std_return = np.std(returns_pct)
        
        if std_return == 0:
            return {
                "anomalies_detected": 0,
                "risk_level": "LOW",
                "message": "No volatility detected - returns are constant",
                "details": {
                    "mean_return": float(mean_return),
                    "std_return": float(std_return)
                }
            }
        
        # Calculate z-scores
        z_scores = (returns_pct - mean_return) / std_return
        
        # Detect anomalies: z-score > 2.5 or < -2.5 (statistical outliers)
        anomaly_threshold = 2.5
        anomalies = np.abs(z_scores) > anomaly_threshold
        
        # Count anomalies
        anomaly_count = int(np.sum(anomalies))
        
        # Get most recent anomaly if any
        recent_anomalies = []
        if anomaly_count > 0:
            # Get last 5 days for recent anomaly check
            recent_z_scores = z_scores[-5:] if len(z_scores) >= 5 else z_scores
            recent_returns = returns_pct[-5:] if len(returns_pct) >= 5 else returns_pct
            
            for i, (z_score, ret) in enumerate(zip(recent_z_scores, recent_returns)):
                if abs(z_score) > anomaly_threshold:
                    recent_anomalies.append({
                        "days_ago": len(recent_z_scores) - i - 1,
                        "z_score": float(z_score),
                        "return_pct": float(ret),
                        "severity": "HIGH" if abs(z_score) > 3.5 else "MEDIUM"
                    })
        
        # Determine risk level based on anomaly count and severity
        if anomaly_count == 0:
            risk_level = "LOW"
            message = "No anomalies detected. Stock returns are within normal statistical range."
        elif anomaly_count <= 2:
            # Check if recent anomalies exist
            if recent_anomalies and any(a['severity'] == 'HIGH' for a in recent_anomalies):
                risk_level = "HIGH"
                message = f"{anomaly_count} anomaly(ies) detected, including recent high-severity events. Exercise caution."
            else:
                risk_level = "MEDIUM"
                message = f"{anomaly_count} anomaly(ies) detected. Monitor stock performance closely."
        elif anomaly_count <= 5:
            risk_level = "MEDIUM"
            message = f"{anomaly_count} anomalies detected. Increased volatility observed. Consider risk management strategies."
        else:
            risk_level = "HIGH"
            message = f"{anomaly_count} anomalies detected. High volatility and unusual patterns observed. High risk conditions."
        
        # Add details about recent anomalies if any
        details = {
            "total_anomalies": anomaly_count,
            "anomaly_threshold_zscore": anomaly_threshold,
            "mean_return_pct": float(mean_return),
            "std_return_pct": float(std_return),
            "recent_anomalies": recent_anomalies[:3] if recent_anomalies else [],  # Last 3 recent anomalies
            "data_period": {
                "start_date": str(df['ds'].iloc[0]) if 'ds' in df.columns else None,
                "end_date": str(df['ds'].iloc[-1]) if 'ds' in df.columns else None,
                "total_days": len(df)
            }
        }
        
        return {
            "anomalies_detected": anomaly_count,
            "risk_level": risk_level,
            "message": message,
            "details": details
        }
        
    except Exception as e:
        # Graceful error handling
        return {
            "anomalies_detected": 0,
            "risk_level": "LOW",
            "message": f"Error during anomaly detection: {str(e)}",
            "details": {
                "error": str(e)
            }
        }


@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_stock_anomalies(request: AnomalyDetectionRequest):
    """
    Detect anomalies in stock returns using z-score statistical analysis
    
    Returns:
        - anomalies_detected: Number of anomalies found
        - risk_level: LOW, MEDIUM, or HIGH
        - message: Human-readable description
    """
    try:
        result = detect_anomalies_zscore(
            ticker=request.ticker,
            use_real_sentiment=request.use_real_sentiment
        )
        
        return AnomalyDetectionResponse(
            ticker=request.ticker.upper(),
            anomalies_detected=result["anomalies_detected"],
            risk_level=result["risk_level"],
            message=result["message"],
            details=result.get("details")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed for {request.ticker}: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for anomaly detection router"""
    return {
        "status": "healthy",
        "service": "anomaly_detection",
        "timestamp": datetime.now().isoformat()
    }









