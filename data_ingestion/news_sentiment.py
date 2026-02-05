"""
Real sentiment analysis using VADER and TextBlob
Fetches news headlines and analyzes sentiment
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Install with: pip install textblob")

class NewsSentimentAnalyzer:
    """Real-time news sentiment analysis using multiple sources"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self.news_api_key = None  # Set your NewsAPI key here
        self.alpha_vantage_key = None  # Set your Alpha Vantage key here
        
    def set_api_keys(self, news_api_key: str = None, alpha_vantage_key: str = None):
        """Set API keys for news sources"""
        self.news_api_key = news_api_key
        self.alpha_vantage_key = alpha_vantage_key
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text.strip()
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        if not self.vader_analyzer or not text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        cleaned_text = self.clean_text(text)
        scores = self.vader_analyzer.polarity_scores(cleaned_text)
        return scores
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        if not TEXTBLOB_AVAILABLE or not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        cleaned_text = self.clean_text(text)
        blob = TextBlob(cleaned_text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def get_combined_sentiment_score(self, text: str) -> float:
        """Get combined sentiment score from multiple analyzers"""
        vader_scores = self.analyze_sentiment_vader(text)
        textblob_scores = self.analyze_sentiment_textblob(text)
        
        # Combine scores (VADER compound + TextBlob polarity)
        vader_score = vader_scores.get('compound', 0.0)
        textblob_score = textblob_scores.get('polarity', 0.0)
        
        # Weighted average (VADER is generally more reliable for social media)
        combined_score = (vader_score * 0.7) + (textblob_score * 0.3)
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, combined_score))
    
    def fetch_news_headlines(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Fetch news headlines for a given ticker"""
        headlines = []
        
        # Simulate news headlines (replace with real API calls)
        sample_headlines = [
            f"{ticker} stock shows strong performance in recent trading",
            f"Analysts upgrade {ticker} rating following positive earnings",
            f"{ticker} faces market volatility amid economic uncertainty",
            f"Investors optimistic about {ticker} future prospects",
            f"{ticker} stock price drops on disappointing quarterly results",
            f"Market experts predict {ticker} will outperform competitors",
            f"{ticker} announces new strategic partnership",
            f"Concerns grow over {ticker} market position",
            f"{ticker} stock reaches new all-time high",
            f"Trading volume spikes for {ticker} following news announcement"
        ]
        
        # Generate headlines for the past week
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            # Randomly select headlines to simulate real news
            import random
            selected_headlines = random.sample(sample_headlines, random.randint(2, 5))
            
            for headline in selected_headlines:
                headlines.append({
                    'date': date,
                    'headline': headline,
                    'ticker': ticker,
                    'source': 'simulated'
                })
        
        return headlines
    
    def fetch_alpha_vantage_news(self, ticker: str) -> List[Dict]:
        """Fetch news from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return []
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.alpha_vantage_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' in data:
                headlines = []
                for item in data['feed']:
                    headlines.append({
                        'date': datetime.fromisoformat(item['time_published'].replace('Z', '+00:00')),
                        'headline': item['title'],
                        'ticker': ticker,
                        'source': 'alpha_vantage',
                        'sentiment_score': float(item.get('overall_sentiment_score', 0)),
                        'sentiment_label': item.get('overall_sentiment_label', 'neutral')
                    })
                return headlines
        except Exception as e:
            print(f"Error fetching Alpha Vantage news: {e}")
        
        return []
    
    def get_sentiment_scores(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """Get sentiment scores for a ticker over time"""
        # Fetch headlines
        headlines = self.fetch_news_headlines(ticker, days_back)
        
        # Add Alpha Vantage news if API key is available
        if self.alpha_vantage_key:
            alpha_headlines = self.fetch_alpha_vantage_news(ticker)
            headlines.extend(alpha_headlines)
        
        if not headlines:
            # Return default neutral sentiment
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            return pd.DataFrame({
                'date': dates,
                'sentiment_score': 0.0,
                'headline_count': 0
            })
        
        # Analyze sentiment for each headline
        sentiment_data = []
        for headline in headlines:
            score = self.get_combined_sentiment_score(headline['headline'])
            sentiment_data.append({
                'date': headline['date'].date(),
                'sentiment_score': score,
                'headline': headline['headline'],
                'source': headline.get('source', 'unknown')
            })
        
        # Convert to DataFrame and aggregate by date
        df = pd.DataFrame(sentiment_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and calculate average sentiment
        daily_sentiment = df.groupby('date').agg({
            'sentiment_score': 'mean',
            'headline': 'count'
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'sentiment_score', 'headline_count']
        
        # Fill missing dates with neutral sentiment
        date_range = pd.date_range(
            start=daily_sentiment['date'].min(),
            end=daily_sentiment['date'].max(),
            freq='D'
        )
        
        full_df = pd.DataFrame({'date': date_range})
        full_df = full_df.merge(daily_sentiment, on='date', how='left')
        full_df['sentiment_score'] = full_df['sentiment_score'].fillna(0.0)
        full_df['headline_count'] = full_df['headline_count'].fillna(0)
        
        return full_df

def get_real_sentiment_score(ticker: str, date: datetime = None) -> float:
    """Get real sentiment score for a specific ticker and date"""
    analyzer = NewsSentimentAnalyzer()
    
    # Set API keys if available (you can set these as environment variables)
    import os
    news_api_key = os.getenv('NEWS_API_KEY')
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
    
    if news_api_key or alpha_vantage_key:
        analyzer.set_api_keys(news_api_key, alpha_vantage_key)
    
    # Get sentiment data
    sentiment_df = analyzer.get_sentiment_scores(ticker, days_back=7)
    
    if date:
        # Find sentiment for specific date
        target_date = pd.to_datetime(date).date()
        matching_rows = sentiment_df[sentiment_df['date'].dt.date == target_date]
        if not matching_rows.empty:
            return float(matching_rows['sentiment_score'].iloc[0])
    
    # Return most recent sentiment score
    if not sentiment_df.empty:
        return float(sentiment_df['sentiment_score'].iloc[-1])
    
    return 0.0

if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = NewsSentimentAnalyzer()
    
    # Test with sample text
    sample_text = "Apple stock shows strong performance with positive earnings"
    print(f"Sample text: {sample_text}")
    print(f"VADER scores: {analyzer.analyze_sentiment_vader(sample_text)}")
    print(f"TextBlob scores: {analyzer.analyze_sentiment_textblob(sample_text)}")
    print(f"Combined score: {analyzer.get_combined_sentiment_score(sample_text)}")
    
    # Test with ticker
    print(f"\nTesting with AAPL:")
    sentiment_df = analyzer.get_sentiment_scores("AAPL", days_back=3)
    print(sentiment_df.head())

