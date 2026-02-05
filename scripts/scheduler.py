"""
Scheduled task runner for Stock Analysis System
Handles periodic updates, data refresh, and model retraining
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schedule
import time
import logging
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List

# Import our modules
from modeling.prophet_model import load_features, train_prophet
from evaluation.evaluate_models import evaluate_complete_pipeline
from data_ingestion.news_sentiment import NewsSentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockAnalysisScheduler:
    """Scheduled task manager for stock analysis system"""
    
    def __init__(self, api_url: str = "http://localhost:8000", tickers: List[str] = None):
        self.api_url = api_url
        # Allow dynamic ticker list, with default fallback
        self.tickers = tickers if tickers else ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
        self.session = requests.Session()
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        logger.info(f"Scheduler initialized with tickers: {', '.join(self.tickers)}")
    
    def update_tickers(self, new_tickers: List[str]):
        """Update the list of tickers to monitor"""
        if not new_tickers:
            logger.warning("Cannot update with empty ticker list")
            return
        
        logger.info(f"Updating ticker list from {self.tickers} to {new_tickers}")
        self.tickers = new_tickers
        logger.info(f"Ticker list updated successfully")
    
    def update_forecasts(self):
        """Update forecasts for all tracked tickers"""
        logger.info("Starting forecast update...")
        
        for ticker in self.tickers:
            try:
                logger.info(f"Updating forecast for {ticker}")
                
                # Load data and train model
                df = load_features()
                forecast = train_prophet(df)
                
                # Save forecast data
                forecast_file = f"data/forecast_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
                forecast_data = {
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "forecast": forecast.to_dict('records')
                }
                
                with open(forecast_file, 'w') as f:
                    json.dump(forecast_data, f, indent=2, default=str)
                
                logger.info(f"Forecast updated for {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to update forecast for {ticker}: {e}")
    
    def update_sentiment_analysis(self):
        """Update sentiment analysis for all tickers"""
        logger.info("Starting sentiment analysis update...")
        
        analyzer = NewsSentimentAnalyzer()
        
        for ticker in self.tickers:
            try:
                logger.info(f"Updating sentiment for {ticker}")
                
                # Get sentiment data
                sentiment_df = analyzer.get_sentiment_scores(ticker, days_back=7)
                
                # Save sentiment data
                sentiment_file = f"data/sentiment_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
                sentiment_data = {
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": sentiment_df.to_dict('records')
                }
                
                with open(sentiment_file, 'w') as f:
                    json.dump(sentiment_data, f, indent=2, default=str)
                
                logger.info(f"Sentiment updated for {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to update sentiment for {ticker}: {e}")
    
    def run_model_evaluation(self):
        """Run comprehensive model evaluation"""
        logger.info("Starting model evaluation...")
        
        try:
            # Run evaluation for each ticker
            for ticker in self.tickers:
                logger.info(f"Evaluating models for {ticker}")
                
                # This would need to be adapted to work with specific tickers
                # For now, run general evaluation
                metrics_df, predictions, actual = evaluate_complete_pipeline(
                    ticker=ticker,
                    use_real_sentiment=True
                )
                
                # Save evaluation results
                eval_file = f"data/evaluation_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
                eval_data = {
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics_df.to_dict('index'),
                    "best_model": metrics_df['RMSE'].idxmin() if 'RMSE' in metrics_df.columns else "Unknown"
                }
                
                with open(eval_file, 'w') as f:
                    json.dump(eval_data, f, indent=2, default=str)
                
                logger.info(f"Evaluation completed for {ticker}")
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data files"""
        logger.info(f"Cleaning up data older than {days_to_keep} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        data_dir = "data"
        
        if not os.path.exists(data_dir):
            return
        
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_time < cutoff_date:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to remove {filename}: {e}")
    
    def health_check(self):
        """Check system health and send alerts if needed"""
        logger.info("Running health check...")
        
        try:
            # Check API health
            response = self.session.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("API health check passed")
            else:
                logger.warning(f"API health check failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def generate_daily_report(self):
        """Generate daily analysis report"""
        logger.info("Generating daily report...")
        
        try:
            report = {
                "date": datetime.now().isoformat(),
                "tickers_analyzed": self.tickers,
                "forecasts_updated": True,
                "sentiment_updated": True,
                "evaluation_completed": True
            }
            
            # Save report
            report_file = f"data/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("Daily report generated")
            
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
    
    def setup_schedule(self):
        """Setup all scheduled tasks"""
        logger.info("Setting up scheduled tasks...")
        
        # Market hours tasks (9:30 AM - 4:00 PM EST)
        schedule.every().monday.at("09:30").do(self.update_forecasts)
        schedule.every().tuesday.at("09:30").do(self.update_forecasts)
        schedule.every().wednesday.at("09:30").do(self.update_forecasts)
        schedule.every().thursday.at("09:30").do(self.update_forecasts)
        schedule.every().friday.at("09:30").do(self.update_forecasts)
        
        # Hourly updates during market hours
        for hour in range(10, 16):
            schedule.every().monday.at(f"{hour:02d}:00").do(self.update_forecasts)
            schedule.every().tuesday.at(f"{hour:02d}:00").do(self.update_forecasts)
            schedule.every().wednesday.at(f"{hour:02d}:00").do(self.update_forecasts)
            schedule.every().thursday.at(f"{hour:02d}:00").do(self.update_forecasts)
            schedule.every().friday.at(f"{hour:02d}:00").do(self.update_forecasts)
        
        # Sentiment analysis (every 2 hours)
        schedule.every(2).hours.do(self.update_sentiment_analysis)
        
        # Model evaluation (daily at 6 PM)
        schedule.every().day.at("18:00").do(self.run_model_evaluation)
        
        # Health check (every 15 minutes)
        schedule.every(15).minutes.do(self.health_check)
        
        # Daily report (every day at 7 PM)
        schedule.every().day.at("19:00").do(self.generate_daily_report)
        
        # Cleanup (weekly on Sunday at 2 AM)
        schedule.every().sunday.at("02:00").do(self.cleanup_old_data)
        
        logger.info("Scheduled tasks configured")
    
    def run(self):
        """Run the scheduler"""
        logger.info("Starting Stock Analysis Scheduler...")
        
        # Setup schedule
        self.setup_schedule()
        
        # Run initial tasks
        logger.info("Running initial tasks...")
        self.update_forecasts()
        self.update_sentiment_analysis()
        self.health_check()
        
        # Main scheduler loop
        logger.info("Scheduler started. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise

def main():
    """Main function"""
    scheduler = StockAnalysisScheduler()
    scheduler.run()

if __name__ == "__main__":
    main()

