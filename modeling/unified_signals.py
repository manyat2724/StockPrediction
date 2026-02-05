"""
Unified trading signal logic - used by ALL components
Ensures consistency across main card, sentiment box, trading signals tab, and final call
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

def get_final_recommendation(
    forecast: pd.DataFrame,
    df: pd.DataFrame,
    current_price: Optional[float] = None,
    sentiment: Optional[float] = None,
    composite_score: Optional[float] = None,
    buy_signals_count: Optional[int] = None,
    sell_signals_count: Optional[int] = None,
    total_signals_count: Optional[int] = None,
    trend_direction: Optional[str] = None
) -> Dict:
    """
    Centralized function to compute final trading recommendation
    Uses decision framework that never contradicts composite score, signals, or trend
    
    Decision Framework:
    1. Base recommendation on expected_change (±2% thresholds)
    2. Modify using composite_score (>=75 upgrades, <=30 downgrades)
    3. Incorporate trading signal majority (buy/sell ratio > 0.60)
    4. Include trend direction (Bullish/Bearish shifts)
    5. Ensure no contradictions
    """
    if len(forecast) < 2:
        return {
            "recommendation": "HOLD",
            "confidence": 0.0,
            "expected_change": 0.0,
            "reasoning": "Insufficient forecast data"
        }
    
    # Get current price
    if current_price is None:
        if 'close_price' in df.columns:
            current_price = float(df['close_price'].iloc[-1])
        elif 'y' in df.columns:
            current_price = float(df['y'].iloc[-1])
        else:
            current_price = 0.0
    
    # Get predicted price (next period)
    if 'yhat' in forecast.columns:
        predicted_price = float(forecast['yhat'].iloc[-1])
    else:
        predicted_price = current_price
    
    # Calculate expected change percentage
    if current_price > 0:
        expected_change = ((predicted_price - current_price) / current_price) * 100
    else:
        expected_change = 0.0
    
    # Calculate confidence from multiple factors:
    # 1. Model error (confidence interval width)
    # 2. Historical accuracy (if available)
    # 3. Volatility (lower volatility = higher confidence)
    confidence_factors = []
    
    # Factor 1: Confidence interval width
    if 'yhat_upper' in forecast.columns and 'yhat_lower' in forecast.columns:
        confidence_width = float(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])
        confidence_ratio = confidence_width / predicted_price if predicted_price > 0 else 1.0
        interval_confidence = max(0, min(100, (1 - min(confidence_ratio, 1)) * 100))
        confidence_factors.append(interval_confidence)
    
    # Factor 2: Historical accuracy (if available in df)
    if 'Volatility' in df.columns and len(df) > 0:
        recent_volatility = float(df['Volatility'].iloc[-1]) if 'Volatility' in df.columns else 0.0
        # Lower volatility = higher confidence
        # Normalize volatility (assuming typical range 0-0.1)
        vol_confidence = max(0, min(100, (1 - min(recent_volatility * 10, 1)) * 100))
        confidence_factors.append(vol_confidence * 0.3)  # 30% weight
    
    # Factor 3: Model error (if we have historical predictions)
    if len(forecast) > len(df) and len(df) > 10:
        # Compare recent historical predictions vs actual
        hist_len = min(len(df), len(forecast))
        if 'close_price' in df.columns:
            actual_prices = df['close_price'].values[-hist_len:]
            predicted_prices = forecast['yhat'].values[-hist_len:]
            errors = np.abs(actual_prices - predicted_prices) / actual_prices
            avg_error = float(np.mean(errors))
            # Lower error = higher confidence
            error_confidence = max(0, min(100, (1 - min(avg_error * 10, 1)) * 100))
            confidence_factors.append(error_confidence * 0.2)  # 20% weight
    
    # Combine factors (weighted average)
    if confidence_factors:
        confidence = sum(confidence_factors) / len(confidence_factors) if len(confidence_factors) > 1 else confidence_factors[0]
    else:
        confidence = 50.0  # Default if no factors available
    
    # Get sentiment if not provided
    if sentiment is None:
        if 'Sentiment' in df.columns:
            sentiment = float(df['Sentiment'].iloc[-1])
        else:
            sentiment = 0.0
    
    # ========== STEP 0: Check for unanimous signals (override base logic) ==========
    # FIX: If ALL signals are buy (100%), force STRONG_BUY regardless of expected_change
    # If ALL signals are sell (100%), force STRONG_SELL regardless of expected_change
    if buy_signals_count is not None and sell_signals_count is not None and total_signals_count is not None and total_signals_count > 0:
        buy_signals_ratio = buy_signals_count / total_signals_count
        sell_signals_ratio = sell_signals_count / total_signals_count
        
        # If 100% buy signals, force STRONG_BUY
        if buy_signals_ratio >= 1.0 and sell_signals_count == 0:
            return {
                "recommendation": "STRONG_BUY",
                "confidence": round(confidence, 1),
                "expected_change": round(expected_change, 2),
                "predicted_price": round(predicted_price, 2),
                "current_price": round(current_price, 2),
                "sentiment": round(sentiment, 3),
                "composite_score": round(composite_score, 1) if composite_score is not None else 75.0,
                "reasoning": f"All {total_signals_count} signals are BUY (100%) → STRONG_BUY recommendation"
            }
        # If 100% sell signals, force STRONG_SELL
        elif sell_signals_ratio >= 1.0 and buy_signals_count == 0:
            return {
                "recommendation": "STRONG_SELL",
                "confidence": round(confidence, 1),
                "expected_change": round(expected_change, 2),
                "predicted_price": round(predicted_price, 2),
                "current_price": round(current_price, 2),
                "sentiment": round(sentiment, 3),
                "composite_score": round(composite_score, 1) if composite_score is not None else 25.0,
                "reasoning": f"All {total_signals_count} signals are SELL (100%) → STRONG_SELL recommendation"
            }
    
    # ========== STEP 1: NEW SIGNAL-BASED RECOMMENDATION LOGIC ==========
    # FIX: Use signal_ratio and net_signal to determine base recommendation
    # This ensures HOLD is NOT returned when buy signals significantly outnumber sell signals
    base_recommendation = "HOLD"  # Default
    
    if buy_signals_count is not None and sell_signals_count is not None and total_signals_count is not None and total_signals_count > 0:
        # Compute net_signal and signal_ratio
        net_signal = buy_signals_count - sell_signals_count
        signal_ratio = buy_signals_count / total_signals_count
        
        # NEW RULES: Based on signal_ratio and net_signal
        if signal_ratio >= 0.60 or net_signal >= 10:
            base_recommendation = "BUY"
        elif signal_ratio <= 0.40 or net_signal <= -10:
            base_recommendation = "SELL"
        else:
            base_recommendation = "HOLD"
    else:
        # Fallback to expected_change if signal counts not available
        if abs(expected_change) < 2.0:
            base_recommendation = "HOLD"
        elif expected_change >= 2.0:
            base_recommendation = "BUY"
        else:  # expected_change <= -2.0
            base_recommendation = "SELL"
    
    # Recommendation levels: STRONG_SELL < SELL < HOLD < BUY < STRONG_BUY
    recommendation_levels = ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
    current_level = recommendation_levels.index(base_recommendation)
    
    # ========== STEP 2: Modify using composite_score ==========
    if composite_score is not None:
        if composite_score >= 90:
            # Upgrade 2 levels
            current_level = min(4, current_level + 2)
        elif composite_score >= 80:
            # Upgrade 1 level
            current_level = min(4, current_level + 1)
        elif composite_score <= 30:
            # Downgrade HOLD → SELL
            if base_recommendation == "HOLD":
                current_level = max(0, current_level - 1)  # HOLD → SELL
    
    # ========== STEP 3: Additional signal-based adjustments ==========
    # Further refine based on signal strength
    if buy_signals_count is not None and sell_signals_count is not None and total_signals_count is not None and total_signals_count > 0:
        buy_signals_ratio = buy_signals_count / total_signals_count
        sell_signals_ratio = sell_signals_count / total_signals_count
        net_signal = buy_signals_count - sell_signals_count
        
        # If buy_signals ratio > 80%, shift 2 levels toward BUY
        if buy_signals_ratio > 0.8:
            current_level = min(4, current_level + 2)  # Shift 2 levels toward BUY
        # If sell_signals ratio > 80%, shift 2 levels toward SELL
        elif sell_signals_ratio > 0.8:
            current_level = max(0, current_level - 2)  # Shift 2 levels toward SELL
        # If net_signal is very high (>= 20), upgrade to STRONG_BUY
        elif net_signal >= 20 and current_level >= 3:  # Already BUY or higher
            current_level = 4  # STRONG_BUY
        # If net_signal is very low (<= -20), downgrade to STRONG_SELL
        elif net_signal <= -20 and current_level <= 1:  # Already SELL or lower
            current_level = 0  # STRONG_SELL
    
    # ========== STEP 4: Include trend direction ==========
    if trend_direction is not None:
        trend_upper = str(trend_direction).upper()
        if "BULLISH" in trend_upper or "UP" in trend_upper or "RISING" in trend_upper:
            # Shift one level toward BUY
            current_level = min(4, current_level + 1)
        elif "BEARISH" in trend_upper or "DOWN" in trend_upper or "FALLING" in trend_upper:
            # Shift one level toward SELL
            current_level = max(0, current_level - 1)
    
    # ========== STEP 5: Ensure no contradictions ==========
    # Prevent contradictions:
    # - Never show STRONG_BUY when sell_signals are majority
    if sell_signals_count is not None and buy_signals_count is not None:
        if sell_signals_count > buy_signals_count and current_level >= 3:  # BUY or STRONG_BUY
            current_level = 2  # Force to HOLD
    
    # - Never show HOLD when composite > 80
    if composite_score is not None and composite_score > 80:
        if current_level == 2:  # HOLD
            current_level = 3  # Upgrade to BUY
    
    # - Never show BUY when composite < 50, UNLESS all signals are buy (override)
    if composite_score is not None and composite_score < 50:
        # FIX: Don't force to HOLD if ALL signals are buy (100% buy ratio)
        if buy_signals_count is not None and sell_signals_count is not None and total_signals_count is not None:
            buy_signals_ratio = buy_signals_count / total_signals_count if total_signals_count > 0 else 0
            # Only force to HOLD if not all signals are buy
            if buy_signals_ratio < 1.0 and current_level >= 3:  # BUY or STRONG_BUY
                current_level = 2  # Force to HOLD
        elif current_level >= 3:  # BUY or STRONG_BUY
            current_level = 2  # Force to HOLD
    
    # Get final recommendation
    # FIX: Ensure recommendation is always valid
    try:
        recommendation = recommendation_levels[current_level]
    except (IndexError, ValueError):
        # Fallback to HOLD if level is out of bounds
        recommendation = "HOLD"
        current_level = 2
    
    # FIX: Ensure recommendation is always one of the valid values
    if recommendation not in ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]:
        recommendation = "HOLD"
    
    # FIX: Ensure composite_score is in 0-100 range if provided
    if composite_score is not None:
        composite_score = max(0.0, min(100.0, float(composite_score)))
    else:
        # Default composite_score if not provided (based on confidence and expected_change)
        composite_score = max(0.0, min(100.0, confidence + (abs(expected_change) * 5)))
    
    # FIX: Align confidence with composite_score (they should be related)
    # Confidence should roughly match composite_score (both 0-100)
    # Use weighted average: 60% composite_score + 40% original confidence
    if composite_score is not None:
        aligned_confidence = (composite_score * 0.6) + (confidence * 0.4)
        confidence = max(0.0, min(100.0, aligned_confidence))
    
    # Build reasoning
    reasoning_parts = []
    
    # Add signal-based reasoning
    if buy_signals_count is not None and sell_signals_count is not None and total_signals_count is not None and total_signals_count > 0:
        net_signal = buy_signals_count - sell_signals_count
        signal_ratio = buy_signals_count / total_signals_count
        buy_ratio = (buy_signals_count / total_signals_count * 100) if total_signals_count > 0 else 0
        sell_ratio = (sell_signals_count / total_signals_count * 100) if total_signals_count > 0 else 0
        
        reasoning_parts.append(f"Signals: {buy_signals_count} buy, {sell_signals_count} sell (ratio: {signal_ratio:.2f}, net: {net_signal:+d})")
        reasoning_parts.append(f"Base: {base_recommendation} (expected_change: {expected_change:.2f}%)")
        
        if signal_ratio >= 0.60 or net_signal >= 10:
            reasoning_parts.append(f"Buy signal majority ({buy_ratio:.1f}% or net +{net_signal}) → BUY recommendation")
        elif signal_ratio <= 0.40 or net_signal <= -10:
            reasoning_parts.append(f"Sell signal majority ({sell_ratio:.1f}% or net {net_signal}) → SELL recommendation")
    else:
        reasoning_parts.append(f"Base: {base_recommendation} (expected_change: {expected_change:.2f}%)")
    
    if composite_score is not None:
        reasoning_parts.append(f"Composite Score: {composite_score:.1f}/100")
        if composite_score >= 75:
            reasoning_parts.append("High composite score → upgraded recommendation")
        elif composite_score <= 30:
            reasoning_parts.append("Low composite score → downgraded recommendation")
    
    if trend_direction is not None:
        reasoning_parts.append(f"Trend: {trend_direction} → adjusted recommendation")
    
    reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Based on forecast analysis"
    
    return {
        "recommendation": recommendation,  # Always valid: STRONG_SELL/SELL/HOLD/BUY/STRONG_BUY
        "confidence": round(confidence, 1),  # Aligned with composite_score, 0-100
        "expected_change": round(expected_change, 2),
        "predicted_price": round(predicted_price, 2),
        "current_price": round(current_price, 2),
        "sentiment": round(sentiment, 3),
        "composite_score": round(composite_score, 1),  # Always 0-100, always present
        "reasoning": reasoning
    }

def get_signal_strength(recommendation: str, expected_change: float, confidence: float) -> float:
    """Calculate signal strength (0-100) based on recommendation"""
    base_strength = {
        "STRONG_BUY": 90,
        "BUY": 70,
        "HOLD": 50,
        "SELL": 30,
        "STRONG_SELL": 10
    }.get(recommendation, 50)
    
    # Adjust based on expected change magnitude
    change_factor = min(abs(expected_change) / 2.0, 1.0)  # Max at 2% change
    
    # Adjust based on confidence
    confidence_factor = confidence / 100.0
    
    # Combine factors
    strength = base_strength * (0.5 + 0.3 * change_factor + 0.2 * confidence_factor)
    
    return max(0, min(100, strength))

def format_recommendation_for_ui(recommendation_data: Dict) -> Dict:
    """Format recommendation data for UI display"""
    rec = recommendation_data["recommendation"]
    
    # Color mapping
    color_map = {
        "STRONG_BUY": "#52c41a",  # Green
        "BUY": "#73d13d",  # Light green
        "HOLD": "#faad14",  # Yellow/Orange
        "SELL": "#ff7875",  # Light red
        "STRONG_SELL": "#ff4d4f"  # Red
    }
    
    # Icon mapping
    icon_map = {
        "STRONG_BUY": "📈",
        "BUY": "📊",
        "HOLD": "⏸️",
        "SELL": "📉",
        "STRONG_SELL": "🔻"
    }
    
    return {
        **recommendation_data,
        "color": color_map.get(rec, "#faad14"),
        "icon": icon_map.get(rec, "⏸️"),
        "strength": get_signal_strength(
            rec,
            recommendation_data["expected_change"],
            recommendation_data["confidence"]
        )
    }

