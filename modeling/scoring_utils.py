"""
Unified scoring utilities for composite score calculation and label mapping.

This module is the SINGLE SOURCE OF TRUTH for:
- Composite score calculation (0-100 range)
- Recommendation label mapping (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)

All composite scores throughout the application must use these functions to ensure consistency.
"""

from typing import Dict


def compute_composite_score(
    prophet_score: float,
    sentiment_score: float,
    technical_score: float,
    risk_score: float,
) -> float:
    """
    Computes a unified composite score between 0 and 100.
    
    This is the SINGLE SOURCE OF TRUTH for composite score calculation.
    All inputs are normalized to 0-100 range before calculation.
    Returns a float rounded to 1 decimal, clamped to 0-100.
    
    Args:
        prophet_score: Model confidence/forecast score (will be normalized to 0-100)
        sentiment_score: News sentiment score (will be normalized to 0-100)
        technical_score: Technical signal strength (will be normalized to 0-100)
        risk_score: Risk level score (will be normalized to 0-100, higher = safer)
    
    Returns:
        Composite score between 0 and 100 (rounded to 1 decimal)
    
    Weights:
        - Prophet: 40%
        - Sentiment: 20%
        - Technical: 25%
        - Risk: 15%
    """
    weights = {
        "prophet": 0.40,
        "sentiment": 0.20,
        "technical": 0.25,
        "risk": 0.15,
    }
    
    def clamp(x: float) -> float:
        """Clamp value to 0-100 range - CRITICAL to prevent values above 100"""
        return max(0.0, min(100.0, float(x)))
    
    # CRITICAL: Normalize and clamp all inputs to 0-100 range BEFORE calculation
    # This prevents any composite score from exceeding 100
    prophet_score = clamp(float(prophet_score))
    sentiment_score = clamp(float(sentiment_score))
    technical_score = clamp(float(technical_score))
    risk_score = clamp(float(risk_score))
    
    # Calculate weighted average
    final_score = (
        prophet_score * weights["prophet"] +
        sentiment_score * weights["sentiment"] +
        technical_score * weights["technical"] +
        risk_score * weights["risk"]
    )
    
    # Final clamp to ensure 0-100 range (double safety check)
    final_score = clamp(final_score)
    
    # Round to 1 decimal
    return round(final_score, 1)


def map_score_to_label(score: float, expected_change_pct: float) -> str:
    """
    Map a 0-100 composite score plus expected_change_pct to a recommendation label.
    
    This is the SINGLE SOURCE OF TRUTH for recommendation label mapping.
    Both the top dashboard and Final Call bottom card must use this function.
    
    Rules:
    - Neutral zone: -0.5% to +0.5% → ALWAYS returns HOLD
    - Strong negative change (e.g., -30%, -50%, -90%) → NEVER returns HOLD
    - Symmetric logic for up & down trends
    - Thresholds: >= 70 (Strong), >= 55 (Buy/Sell), >= 45 (Hold), < 45 (Hold if neutral, else Sell)
    
    Args:
        score: Composite score (0-100, will be clamped)
        expected_change_pct: Expected price change percentage (can be negative)
    
    Returns:
        Recommendation label: STRONG_BUY, BUY, HOLD, SELL, or STRONG_SELL
        (AVOID label is removed - not supported by UI)
    """
    # Clamp score to 0-100 range
    score = max(0.0, min(100.0, float(score)))
    expected_change_pct = float(expected_change_pct)
    
    # Determine direction with neutral zone
    # Neutral zone: -0.5% to +0.5%
    if expected_change_pct > 0.5:
        direction = "up"
    elif expected_change_pct < -0.5:
        direction = "down"
    else:
        direction = "neutral"
    
    # If direction is neutral → ALWAYS return HOLD
    if direction == "neutral":
        return "HOLD"
    
    # For non-neutral directions, apply symmetric mapping logic
    # Thresholds: >= 70 (Strong), >= 55 (Buy/Sell), >= 45 (Hold), < 45 (Hold if neutral, else Sell)
    
    if score >= 70:
        # Strong recommendation
        return "STRONG_BUY" if direction == "up" else "STRONG_SELL"
    elif score >= 55:
        # Buy/Sell recommendation
        return "BUY" if direction == "up" else "SELL"
    elif score >= 45:
        # Hold zone - but check for strong negative change
        # If strongly negative (e.g., -30%, -50%, -90%), NEVER return HOLD
        if direction == "down" and expected_change_pct < -5.0:
            # Strong negative change → downgrade to SELL
            return "SELL"
        else:
            return "HOLD"
    else:
        # Score < 45: Hold if neutral, otherwise Sell
        # Since we already handled neutral above, this must be down or up
        if direction == "down":
            # Negative trend with low score → SELL
            return "SELL"
        else:
            # Positive trend with low score → still HOLD (conservative)
            return "HOLD"


def normalize_sentiment_to_score(sentiment_raw: float) -> float:
    """
    Normalize sentiment from [-1, 1] range to [0, 100] score.
    
    Args:
        sentiment_raw: Raw sentiment value (typically -1 to 1)
    
    Returns:
        Normalized sentiment score (0-100, clamped)
    """
    # Clamp to [-1, 1] range first
    sentiment_raw = max(-1.0, min(1.0, float(sentiment_raw)))
    # Normalize to 0-100
    normalized = ((sentiment_raw + 1.0) / 2.0) * 100.0
    # Final clamp to ensure 0-100
    return max(0.0, min(100.0, normalized))


def normalize_trend_to_score(trend_direction: str) -> float:
    """
    Map trend direction to a 0-100 score.
    
    Args:
        trend_direction: Trend direction string (Bullish, Neutral, Bearish)
    
    Returns:
        Trend score (0-100, clamped)
    """
    trend_scores = {
        'Bullish': 80.0,
        'Neutral': 50.0,
        'Bearish': 20.0
    }
    score = float(trend_scores.get(trend_direction, 50.0))
    # Clamp to 0-100
    return max(0.0, min(100.0, score))


def normalize_risk_to_score(risk_level: str) -> float:
    """
    Map risk level to a 0-100 score (inverse: lower risk = higher score).
    
    Args:
        risk_level: Risk level string (LOW, MEDIUM, HIGH)
    
    Returns:
        Risk score (0-100, where 100 = safest, clamped)
    """
    risk_scores = {
        'LOW': 90.0,
        'MEDIUM': 50.0,
        'HIGH': 10.0
    }
    score = float(risk_scores.get(risk_level, 50.0))
    # Clamp to 0-100
    return max(0.0, min(100.0, score))
