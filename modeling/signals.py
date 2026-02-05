"""
Automated signal generation for algorithmic trading
Risk management through anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from modeling.unified_signals import get_final_recommendation, format_recommendation_for_ui

def generate_trading_signals(df: pd.DataFrame, forecast: pd.DataFrame) -> Dict:
    """
    Generate automated trading signals based on return predictions
    Uses return-based decisions: BUY if predicted_return > +0.5% and confidence > 60%
    SELL if predicted_return < -0.5% and confidence > 60%, HOLD otherwise
    """
    signals = []
    signal_strength = []
    
    if len(forecast) < 2:
        return {"signals": [], "summary": {}}
    
    # Get predicted returns (use yhat_return if available, otherwise calculate from prices)
    if 'yhat_return' in forecast.columns:
        predicted_returns = forecast['yhat_return']
    else:
        # Calculate returns from predicted prices
        predicted_returns = forecast['yhat'].pct_change().fillna(0)
    
    # Calculate volatility
    volatility = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 0
    
    # Get sentiment if available
    sentiment = df['Sentiment'].iloc[-1] if 'Sentiment' in df.columns else 0
    
    # FIX: Limit to maximum 60 signals for performance and UI display
    max_signals = min(len(forecast), 60)
    
    for i in range(max_signals):
        # Get predicted return for this period
        if i == 0:
            # For first period, use the return directly
            predicted_return = float(predicted_returns.iloc[i]) if i < len(predicted_returns) else 0
        else:
            # For subsequent periods, use the return value
            predicted_return = float(predicted_returns.iloc[i]) if i < len(predicted_returns) else 0
        
        # Calculate confidence from confidence interval width
        confidence_width = forecast.iloc[i]['yhat_upper'] - forecast.iloc[i]['yhat_lower']
        predicted_price = forecast.iloc[i]['yhat']
        confidence_ratio = confidence_width / abs(predicted_price) if abs(predicted_price) > 0 else 1
        confidence = (1 - min(confidence_ratio, 1)) * 100  # Convert to percentage
        
        # Return-based signal generation logic
        # BUY if predicted_return > +0.5% and confidence > 60%
        # SELL if predicted_return < -0.5% and confidence > 60%
        # HOLD otherwise
        predicted_return_pct = predicted_return * 100  # Convert to percentage
        
        if predicted_return_pct > 0.5 and confidence > 60:
            if predicted_return_pct > 2.0 and confidence > 80:
                signal = "STRONG_BUY"
                strength = min(100, (predicted_return_pct * 20 + confidence * 0.5))
            else:
                signal = "BUY"
                strength = min(100, (predicted_return_pct * 15 + confidence * 0.4))
        elif predicted_return_pct < -0.5 and confidence > 60:
            if predicted_return_pct < -2.0 and confidence > 80:
                signal = "STRONG_SELL"
                strength = min(100, (abs(predicted_return_pct) * 20 + confidence * 0.5))
            else:
                signal = "SELL"
                strength = min(100, (abs(predicted_return_pct) * 15 + confidence * 0.4))
        else:
            signal = "HOLD"
            strength = max(0, 50 - abs(predicted_return_pct * 10))
        
        signals.append({
            "date": forecast.iloc[i]['ds'].isoformat() if hasattr(forecast.iloc[i]['ds'], 'isoformat') else str(forecast.iloc[i]['ds']),
            "signal": signal,
            "strength": float(max(0, min(100, strength))),
            "predicted_change": float(predicted_return_pct),
            "confidence": float(confidence)
        })
        signal_strength.append(strength)
    
    # Summary statistics
    buy_signals = sum(1 for s in signals if 'BUY' in s['signal'])
    sell_signals = sum(1 for s in signals if 'SELL' in s['signal'])
    hold_signals = sum(1 for s in signals if s['signal'] == 'HOLD')
    
    avg_strength = np.mean(signal_strength) if signal_strength else 50
    
    # Use unified recommendation function for consistency
    unified_rec = get_final_recommendation(forecast, df)
    
    return {
        "signals": signals,
        "summary": {
            "total_signals": len(signals),
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "average_strength": float(avg_strength),
            "recommendation": unified_rec["recommendation"],  # Use unified recommendation
            "recommendation_data": format_recommendation_for_ui(unified_rec)
        }
    }

def detect_anomalies(df: pd.DataFrame, forecast: pd.DataFrame) -> Dict:
    """
    Robust anomaly detection for risk management
    Detects: sudden price jumps/drops, high volatility, forecast deviations, outlier behavior
    """
    anomalies = []
    
    if len(df) < 10:
        return {
            "anomalies": [],
            "risk_level": "LOW",
            "anomaly_count": 0,
            "message": "Insufficient data for anomaly detection"
        }
    
    # Get price data (use close_price if available, otherwise y)
    if 'close_price' in df.columns:
        prices = df['close_price'].values
        current_price = float(df['close_price'].iloc[-1])
    else:
        prices = df['y'].values
        current_price = float(df['y'].iloc[-1])
    
    if len(prices) < 2:
        return {
            "anomalies": [],
            "risk_level": "LOW",
            "anomaly_count": 0,
            "message": "Insufficient price data"
        }
    
    # ========== CHECK 1: Sudden Price Jumps/Drops ==========
    # Calculate daily price changes
    price_changes = np.diff(prices)
    price_change_pct = (price_changes / prices[:-1]) * 100  # Percentage changes
    
    if len(price_change_pct) >= 10:
        mean_change = np.mean(price_changes)
        std_change = np.std(price_changes)
        
        if std_change > 0:
            # Get most recent price change
            recent_change = price_changes[-1]
            recent_change_pct = price_change_pct[-1]
            
            # Check if change exceeds threshold: mean + 2.5 * std
            threshold = mean_change + 2.5 * std_change
            threshold_pct = np.mean(price_change_pct) + 2.5 * np.std(price_change_pct)
            
            if abs(recent_change) > abs(threshold) or abs(recent_change_pct) > abs(threshold_pct):
                severity = "HIGH" if abs(recent_change_pct) > 5.0 else "MEDIUM"
                anomalies.append({
                    "type": "PRICE_JUMP" if recent_change > 0 else "PRICE_DROP",
                    "severity": severity,
                    "description": f"Sudden price {'jump' if recent_change > 0 else 'drop'} of {recent_change_pct:.2f}% detected. Expected range: ±{abs(threshold_pct):.2f}%",
                    "current_change": float(recent_change_pct),
                    "threshold": float(threshold_pct),
                    "current_price": float(current_price)
                })
    
    # ========== CHECK 2: High Volatility Events ==========
    if 'Volatility' in df.columns and len(df) >= 20:
        recent_vol = df['Volatility'].tail(20).values
        mean_vol = np.mean(recent_vol)
        std_vol = np.std(recent_vol)
        
        if std_vol > 0:
            current_vol = float(df['Volatility'].iloc[-1])
            # Check if volatility > mean + 2 * std
            vol_threshold = mean_vol + 2 * std_vol
            
            if current_vol > vol_threshold:
                vol_z_score = (current_vol - mean_vol) / std_vol
                severity = "HIGH" if vol_z_score > 3 else "MEDIUM"
                anomalies.append({
                    "type": "VOLATILITY_SPIKE",
                    "severity": severity,
                    "description": f"High volatility detected: {current_vol:.4f} (threshold: {vol_threshold:.4f}, {vol_z_score:.2f} std dev above mean)",
                    "current_volatility": float(current_vol),
                    "average_volatility": float(mean_vol),
                    "threshold": float(vol_threshold)
                })
    
    # ========== CHECK 3: Forecast Deviation from Actual ==========
    if len(forecast) > 0 and len(df) > 0:
        # Get overlapping period between forecast and actual data
        min_len = min(len(forecast), len(df))
        
        if min_len >= 10:
            # Get actual prices for comparison
            actual_prices = prices[-min_len:]
            predicted_prices = forecast['yhat'].values[-min_len:]
            
            # Calculate prediction errors
            prediction_errors = np.abs(actual_prices - predicted_prices)
            error_pct = (prediction_errors / actual_prices) * 100
            
            # Calculate model error statistics
            mean_error = np.mean(prediction_errors)
            std_error = np.std(prediction_errors)
            mean_error_pct = np.mean(error_pct)
            std_error_pct = np.std(error_pct)
            
            if std_error > 0:
                # Check most recent prediction error
                recent_error = prediction_errors[-1]
                recent_error_pct = error_pct[-1]
                
                # Threshold: mean + 2.5 * std
                error_threshold = mean_error + 2.5 * std_error
                error_threshold_pct = mean_error_pct + 2.5 * std_error_pct
                
                if recent_error > error_threshold or recent_error_pct > error_threshold_pct:
                    severity = "HIGH" if recent_error_pct > 5.0 else "MEDIUM"
                    anomalies.append({
                        "type": "FORECAST_DEVIATION",
                        "severity": severity,
                        "description": f"Forecast deviates significantly from actual: error of {recent_error_pct:.2f}% (threshold: {error_threshold_pct:.2f}%)",
                        "actual_price": float(actual_prices[-1]),
                        "predicted_price": float(predicted_prices[-1]),
                        "error_percentage": float(recent_error_pct),
                        "threshold": float(error_threshold_pct)
                    })
    
    # ========== CHECK 4: Predicted Price Outside Prophet Bounds ==========
    if len(forecast) > 0:
        # Get the most recent forecast point
        latest_forecast = forecast.iloc[-1]
        predicted_price = float(latest_forecast['yhat'])
        upper_bound = float(latest_forecast['yhat_upper'])
        lower_bound = float(latest_forecast['yhat_lower'])
        
        # Check if predicted price is outside bounds (shouldn't happen, but indicates model uncertainty)
        if predicted_price > upper_bound or predicted_price < lower_bound:
            anomalies.append({
                "type": "BOUNDS_VIOLATION",
                "severity": "HIGH",
                "description": f"Predicted price ({predicted_price:.2f}) outside Prophet confidence bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                "predicted_price": float(predicted_price),
                "upper_bound": float(upper_bound),
                "lower_bound": float(lower_bound)
            })
        
        # Check if current price is far outside forecast bounds (indicates unexpected movement)
        if current_price > upper_bound * 1.05 or current_price < lower_bound * 0.95:
            deviation_pct = ((current_price - predicted_price) / predicted_price) * 100
            severity = "HIGH" if abs(deviation_pct) > 5.0 else "MEDIUM"
            anomalies.append({
                "type": "PRICE_OUTSIDE_BOUNDS",
                "severity": severity,
                "description": f"Current price ({current_price:.2f}) significantly outside forecast bounds. Deviation: {deviation_pct:.2f}%",
                "current_price": float(current_price),
                "predicted_price": float(predicted_price),
                "upper_bound": float(upper_bound),
                "lower_bound": float(lower_bound),
                "deviation_percentage": float(deviation_pct)
            })
    
    # ========== DETERMINE RISK LEVEL ==========
    high_risk_count = sum(1 for a in anomalies if a['severity'] == 'HIGH')
    medium_risk_count = sum(1 for a in anomalies if a['severity'] == 'MEDIUM')
    
    # Risk level logic:
    # HIGH: Any high severity anomaly OR 3+ medium severity anomalies
    # MEDIUM: 1-2 medium severity anomalies OR 1 high + any medium
    # LOW: No anomalies or only low severity
    if high_risk_count > 0 or medium_risk_count >= 3:
        risk_level = "HIGH"
        message = f"High risk detected: {high_risk_count} high-severity and {medium_risk_count} medium-severity anomalies found."
    elif medium_risk_count >= 1 or (high_risk_count == 0 and len(anomalies) > 0):
        risk_level = "MEDIUM"
        message = f"Moderate risk: {medium_risk_count} medium-severity anomaly(ies) detected."
    else:
        risk_level = "LOW"
        message = "No significant anomalies detected. Market conditions appear normal."
    
    # Create summary message
    if len(anomalies) > 0:
        anomaly_types = [a['type'] for a in anomalies]
        unique_types = list(set(anomaly_types))
        message += f" Detected: {', '.join(unique_types)}."
    
    return {
        "anomalies": anomalies,
        "risk_level": risk_level,
        "anomaly_count": len(anomalies),
        "message": message,
        "high_severity_count": high_risk_count,
        "medium_severity_count": medium_risk_count
    }

def calculate_portfolio_metrics(tickers: List[str], weights: List[float], 
                               price_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Calculate portfolio optimization metrics
    """
    if len(tickers) != len(weights) or abs(sum(weights) - 1.0) > 0.01:
        return {"error": "Invalid portfolio configuration"}
    
    portfolio_returns = []
    portfolio_volatility = []
    valid_tickers = []
    valid_weights = []
    
    for ticker, weight in zip(tickers, weights):
        if ticker not in price_data:
            continue
        
        df = price_data[ticker]
        
        # Check for 'y' column (already processed) or 'Close' column (raw data)
        price_col = 'y' if 'y' in df.columns else 'Close' if 'Close' in df.columns else None
        
        if price_col is None or len(df) < 2:
            continue
        
        # Calculate returns
        returns = df[price_col].pct_change().dropna()
        if len(returns) == 0:
            continue
            
        portfolio_returns.append(returns * weight)
        valid_tickers.append(ticker)
        valid_weights.append(weight)
        
        # Get volatility if available
        if 'Volatility' in df.columns:
            portfolio_volatility.append(df['Volatility'].iloc[-1] * weight)
        else:
            # Calculate volatility from returns
            vol = returns.std() if len(returns) > 0 else 0
            portfolio_volatility.append(vol * weight)
    
    if not portfolio_returns:
        return {"error": "No valid data for portfolio calculation. Please ensure all tickers have valid price data."}
    
    # Align all return series to same index
    try:
        # Get common index
        common_index = portfolio_returns[0].index
        for ret in portfolio_returns[1:]:
            common_index = common_index.intersection(ret.index)
        
        if len(common_index) == 0:
            return {"error": "No overlapping dates for portfolio calculation"}
        
        # Align all returns to common index
        aligned_returns = [ret.loc[common_index] for ret in portfolio_returns]
        
        # Combine returns
        combined_returns = pd.concat(aligned_returns, axis=1).sum(axis=1)
        
        # Calculate metrics
        total_return = float(combined_returns.sum() * 100)  # Convert to percentage
        avg_return = float(combined_returns.mean() * 100)  # Convert to percentage
        portfolio_vol = float(combined_returns.std() * 100)  # Convert to percentage
        sharpe_ratio = float(avg_return / portfolio_vol) if portfolio_vol > 0 else 0
        
        return {
            "total_return": total_return,
            "average_return": avg_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe_ratio,
            "portfolio_volatility": float(sum(portfolio_volatility) * 100) if portfolio_volatility else portfolio_vol,
            "tickers": valid_tickers,
            "weights": valid_weights
        }
    except Exception as e:
        return {"error": f"Error calculating portfolio metrics: {str(e)}"}

