# Quick Reference Guide - Stock Prediction System

## 🎯 **ONE-LINER SUMMARY**
Full-stack stock prediction system using Prophet time series models with return-based forecasting, sentiment analysis, risk management, and paper trading.

---

## 📋 **FILE QUICK REFERENCE**

### **Data Ingestion**
| File | Purpose | Key Function |
|------|---------|--------------|
| `stock_fetch.py` | Get stock data | `fetch_stock_data()` → Returns OHLCV |
| `sentiment.py` | Basic sentiment | `get_sentiment_score()` → VADER score |
| `news_sentiment.py` | News analysis | `get_sentiment_scores()` → Daily sentiment |
| `ticker_loader.py` | Load tickers | `load_tickers()` → U.S. stock list |
| `api/main.py` | REST API | 20+ endpoints for all features |

### **Feature Engineering**
| File | Purpose | Key Function |
|------|---------|--------------|
| `feature.py` | Add indicators | `add_rolling_features()` → RSI, MACD, Volume, MA |

### **Modeling**
| File | Purpose | Key Function |
|------|---------|--------------|
| `prophet_model.py` | Prophet forecasting | `train_prophet()` → Cached, deterministic |
| `xgboost_model.py` | XGBoost alternative | `train_xgboost_model()` → Regression |
| `signals.py` | Trading signals | `generate_trading_signals()` → BUY/SELL/HOLD |
| `signals.py` | Risk management | `detect_anomalies()` → Risk level |
| `unified_signals.py` | Centralized logic | `get_final_recommendation()` → Single source |

### **Evaluation**
| File | Purpose | Key Function |
|------|---------|--------------|
| `metrics.py` | Metrics calculation | `calculate_directional_accuracy()` → % correct |
| `evaluate_models.py` | Full evaluation | `evaluate_complete_pipeline()` → Report |

### **Visualization**
| File | Purpose | Key Function |
|------|---------|--------------|
| `plot_forecast.py` | Charts | `plot_forecast_with_sentiment()` → Matplotlib/Plotly |

### **Frontend**
| File | Purpose | Key Component |
|------|---------|---------------|
| `App.js` | React dashboard | Forecast chart, signals, risk, metrics |

---

## 🔑 **KEY FUNCTIONS TO REMEMBER**

### **Prophet Model** (`prophet_model.py`)
```python
train_prophet(df, periods, ticker, force_retrain)
# - Uses caching (only retrains if data changed)
# - Deterministic (fixed seed 42)
# - Return-based prediction
# - Logistic growth (prevents spikes)
# - Multiple regressors (RSI, MACD, sentiment, volume, MA)
```

### **Anomaly Detection** (`signals.py`)
```python
detect_anomalies(df, forecast)
# Checks:
# 1. Price jumps/drops (>2.5 std dev)
# 2. Volatility spikes (>2 std dev)
# 3. Forecast deviation (>2.5 std dev)
# 4. Price outside bounds
# Returns: Risk level (LOW/MEDIUM/HIGH)
```

### **Unified Recommendation** (`unified_signals.py`)
```python
get_final_recommendation(forecast, df, current_price, sentiment)
# Rules:
# - STRONG_BUY: change >2% AND confidence >70% AND sentiment >0.1
# - BUY: change 1-2% AND confidence >=50%
# - HOLD: change <1% OR low confidence
# - SELL: change -1% to -2%
# - STRONG_SELL: change <-2% AND sentiment <0
```

### **Metrics** (`metrics.py`)
```python
calculate_directional_accuracy(actual, predicted, use_returns)
# - Compares direction (up/down) of predictions vs actuals
# - Returns percentage of correct predictions
# - Key metric for trading systems
```

---

## 🎯 **INTERVIEW ANSWERS**

### **Q: What does your system do?**
**A**: "It's a stock prediction system that forecasts prices using Prophet time series models. The system predicts returns (not prices) for better stability, includes sentiment analysis from news, generates trading signals, detects anomalies for risk management, and provides paper trading capabilities. The frontend is a React dashboard showing forecasts, signals, and metrics."

### **Q: How does Prophet work in your system?**
**A**: "I use Prophet with logistic growth to prevent unrealistic price spikes. The model predicts returns instead of prices, which improves directional accuracy. I add multiple regressors - RSI, MACD, volume change, and sentiment - to help the model learn patterns. The model is cached to disk and only retrains when new data arrives, ensuring deterministic predictions."

### **Q: How do you prevent overfitting?**
**A**: "I use several techniques: 1) Lower changepoint_prior_scale (0.05) for smoother predictions, 2) RobustScaler for outlier resistance, 3) Clip returns to [-10%, +10%], 4) Logistic growth with cap/floor constraints, 5) Normalize all regressors before training."

### **Q: How do you ensure predictions are consistent?**
**A**: "I implemented model caching with data hash validation. The model only retrains if: 1) Model doesn't exist, 2) Data hash changed (new data), 3) Model is from different day. I also use a fixed random seed (42) for deterministic behavior. This ensures the same forecast on every refresh until new data arrives."

### **Q: How does anomaly detection work?**
**A**: "I use 4-factor detection: 1) Sudden price jumps/drops (>2.5 std dev from mean), 2) Volatility spikes (>2 std dev above rolling mean), 3) Forecast deviation (actual vs predicted >2.5 std dev), 4) Price outside Prophet confidence bounds. Based on number and severity of anomalies, I assign risk levels: HIGH (any high-severity or 3+ medium), MEDIUM (1-2 medium), LOW (none)."

### **Q: What's the difference between Prophet and XGBoost?**
**A**: "Prophet is a time series model that handles seasonality and trends well, perfect for stock price forecasting. XGBoost is a gradient boosting model that can learn complex non-linear patterns from features. I use Prophet as the primary model because it's designed for time series, but XGBoost is available as an alternative."

### **Q: How do you measure model performance?**
**A**: "I use multiple metrics: RMSE and MAE for error magnitude, MAPE for percentage error, and most importantly Directional Accuracy - the percentage of times the model correctly predicts if price goes up or down. For trading, directional accuracy is more important than exact price prediction."

### **Q: What technologies did you use?**
**A**: "Backend: Python, FastAPI, Prophet, XGBoost, pandas, numpy. Frontend: React.js, Recharts, Ant Design, Axios. Data: yfinance for stock data, VADER/TextBlob for sentiment. Visualization: Matplotlib, Plotly."

---

## 📊 **DATA FLOW (Simple)**

```
User → Frontend → API → Fetch Data → Add Features → Train Model → 
Generate Forecast → Create Signals → Detect Anomalies → 
Return to Frontend → Display Charts
```

---

## 🎓 **KEY CONCEPTS TO KNOW**

1. **Return-Based Prediction**: Predict percentage changes, not absolute prices
2. **Directional Accuracy**: % of correct up/down predictions (target 80%+)
3. **Model Caching**: Save trained models, only retrain when needed
4. **Deterministic**: Same input always gives same output (fixed seed)
5. **Logistic Growth**: Prevents Prophet from predicting unrealistic values
6. **Regressors**: Additional features (RSI, MACD, etc.) to improve predictions
7. **Anomaly Detection**: Statistical methods to identify unusual patterns
8. **Unified Signals**: Single function for all recommendations (consistency)

---

## 💡 **PROBLEM-SOLVING STORIES**

### **Problem**: Forecasts changed on every refresh
**Solution**: Implemented model caching with data hash validation + fixed random seed

### **Problem**: Unrealistic price spikes (80,000)
**Solution**: Switched to return-based prediction + logistic growth + clipping

### **Problem**: Risk always showed green
**Solution**: Built 4-factor anomaly detection with proper thresholds

### **Problem**: Recommendations conflicted (card said BUY, box said HOLD)
**Solution**: Created unified signal function used by all components

### **Problem**: Ticker dropdown empty
**Solution**: Built ticker loader with multiple fallback layers

---

## 🚀 **DEPLOYMENT READY**

- ✅ REST API with FastAPI
- ✅ React frontend
- ✅ Model persistence (pickle)
- ✅ Error handling
- ✅ Caching mechanisms
- ✅ Background tasks
- ✅ Comprehensive logging

---

**Study this + PROJECT_SUMMARY.md for complete interview preparation!**









