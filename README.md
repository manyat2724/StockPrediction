# 📈 AI-Powered Stock Analysis & Prediction System

A comprehensive real-time stock analysis platform combining machine learning predictions, sentiment analysis, and portfolio optimization in an interactive dashboard.

---

## 🚀 Features

### Core Analysis
- **Price Forecasting**: Prophet & XGBoost models for accurate price predictions
- **Trading Signals**: AI-generated BUY/SELL/HOLD recommendations with confidence scores
- **Sentiment Analysis**: News sentiment analysis using VADER & TextBlob
- **Risk Management**: Anomaly detection and risk level assessment

### Advanced Features
- **Model Benchmarking**: Compare Prophet, XGBoost, Naive, and Moving Average models
- **Backtesting**: Simulate trading strategies on historical data
- **Stock Comparison**: Compare multiple stocks with visualizations
- **Paper Trading**: Practice trading with virtual money
- **Portfolio Optimization**: Sharpe ratio, volatility analysis, and optimal weights
- **Market Insights**: Trend, momentum, and volume analysis
- **Real-time Alerts**: Price and sentiment-based alerts

### Dashboard
- **Dark/Light Mode**: Toggle between themes
- **Dynamic Ticker Management**: Add/remove stocks from UI
- **Interactive Charts**: Recharts-powered visualizations
- **Final Recommendation Panel**: Composite score combining all factors

---

## 📁 Project Structure

```
stockproject/
├── data_ingestion/
│   ├── api/
│   │   └── main.py              # FastAPI backend (all endpoints)
│   ├── stock_fetch.py           # Yahoo Finance data fetching
│   ├── news_sentiment.py        # News sentiment analysis
│   └── sentiment.py             # VADER/TextBlob sentiment
├── modeling/
│   ├── prophet_model.py         # Prophet forecasting model
│   ├── xgboost_model.py         # XGBoost prediction model
│   ├── signals.py               # Trading signal generation
│   └── advanced_analytics.py    # Advanced analysis functions
├── feature_engineering/
│   └── feature.py               # Feature engineering & rolling stats
├── evaluation/
│   ├── evaluate_models.py       # Model evaluation framework
│   └── metrics.py               # Accuracy metrics (MAE, RMSE, etc.)
├── visualization/
│   └── plot_forecast.py         # Matplotlib visualizations
├── scripts/
│   └── scheduler.py             # Background task scheduler
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main React dashboard
│   │   ├── index.js             # Entry point
│   │   └── index.css            # Styles
│   ├── public/
│   │   └── index.html           # HTML template
│   └── package.json             # NPM dependencies
├── start_system.py              # System startup script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

```bash
# Clone/navigate to project
cd stockproject

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# For macOS users (XGBoost requires OpenMP)
brew install libomp
```

### Frontend Setup

```bash
cd frontend
npm install
```

---

## 🚀 Running the Application

### Option 1: Start System Script (Recommended)

```bash
python3 start_system.py
```

This automatically starts both backend and frontend.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd stockproject
python3 -m uvicorn data_ingestion.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd stockproject/frontend
npm start
```

### Access the Application
- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

---

## 📊 API Endpoints

### Forecasting
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forecast` | POST | Get price predictions |
| `/signals` | POST | Get trading signals |
| `/signals-enhanced` | POST | Enhanced signals with explanations |

### Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sentiment` | POST | Sentiment analysis |
| `/anomalies` | POST | Anomaly detection |
| `/evaluate` | POST | Model evaluation |
| `/news` | POST | News summary |

### Portfolio
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio` | POST | Portfolio optimization |
| `/backtest` | POST | Backtest trading strategy |
| `/compare` | POST | Compare multiple stocks |

### Trading
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/final-recommendation` | POST | Comprehensive recommendation |
| `/paper-trade/execute` | POST | Execute paper trade |
| `/paper-trade/account` | GET | Get paper trading account |
| `/trade-recommendation` | POST | Get trade recommendation |

### Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tickers` | GET | Get monitored tickers |
| `/tickers/add` | POST | Add ticker to monitor |
| `/tickers/remove` | DELETE | Remove ticker |
| `/alerts` | GET | Get active alerts |
| `/market-insights` | POST | Get market insights |

---

## 🔧 How It Works

### 1. Data Pipeline
```
Yahoo Finance → Stock Data → Feature Engineering → Model Training → Predictions
                    ↓
             News APIs → Sentiment Analysis → Combined Score
```

### 2. Signal Generation
Signals are generated based on:
- **Predicted Price Change**: >2% → STRONG_BUY, >1% → BUY, <-1% → SELL, <-2% → STRONG_SELL
- **Sentiment Score**: Positive/negative news impact
- **Model Confidence**: Prediction reliability score
- **Risk Level**: Anomaly-based risk assessment

### 3. Final Recommendation
Composite score combining:
- Forecast Model (40%)
- News Sentiment (20%)
- Trading Signals (25%)
- Risk Level (15%)

### 4. Model Comparison
- **Prophet**: Best for real-time predictions, handles seasonality
- **XGBoost**: Good for pattern recognition
- **Naive/MA**: Baseline models (flatline predictions)

---

## 📱 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Price Forecast** | Historical prices + predictions chart |
| **Trading Signals** | AI signals with explanations |
| **Risk Management** | Anomaly detection & risk scoring |
| **Sentiment** | News sentiment analysis |
| **Portfolio** | Multi-stock portfolio optimization |
| **Model Benchmark** | Compare model performance |
| **Alerts** | Price & sentiment alerts |
| **Backtest** | Strategy backtesting results |
| **Compare** | Multi-stock comparison charts |
| **Insights** | Market trend analysis |
| **Paper Trade** | Simulated trading |
| **Final Call** | Comprehensive recommendation |

---

## 🎯 Key Technologies

| Category | Technologies |
|----------|-------------|
| **Backend** | FastAPI, Python, Pandas, NumPy |
| **ML Models** | Prophet, XGBoost, scikit-learn |
| **Sentiment** | VADER, TextBlob |
| **Data Source** | yfinance (Yahoo Finance) |
| **Frontend** | React, Ant Design, Recharts |
| **Visualization** | Matplotlib, Recharts |

---

## ⚠️ Disclaimer

This is an educational project. Stock predictions are inherently uncertain. Do not use this system for actual trading decisions without proper financial advice. Past performance does not guarantee future results.

---

## 📄 License

MIT License - Feel free to use and modify for educational purposes.

---

## 🙏 Acknowledgments

- Yahoo Finance for stock data
- Facebook/Meta for Prophet library
- Ant Design for UI components
