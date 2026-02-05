"""
XGBoost model for stock prediction
Implements regression model with rolling features and sentiment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

def prepare_xgboost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for XGBoost model
    """
    # Create lag features
    df['close_lag_1'] = df['y'].shift(1)
    df['close_lag_2'] = df['y'].shift(2)
    df['close_lag_3'] = df['y'].shift(3)
    
    # Rolling averages
    df['ma_3'] = df['y'].rolling(window=3).mean()
    df['ma_7'] = df['y'].rolling(window=7).mean()
    df['ma_14'] = df['y'].rolling(window=14).mean()
    
    # Rolling volatility
    df['volatility_3'] = df['y'].rolling(window=3).std()
    df['volatility_7'] = df['y'].rolling(window=7).std()
    
    # Price changes
    df['price_change_1'] = df['y'].pct_change(1)
    df['price_change_3'] = df['y'].pct_change(3)
    df['price_change_7'] = df['y'].pct_change(7)
    
    # Sentiment features
    if 'Sentiment' in df.columns:
        df['sentiment_lag_1'] = df['Sentiment'].shift(1)
        df['sentiment_ma_3'] = df['Sentiment'].rolling(window=3).mean()
        df['sentiment_ma_7'] = df['Sentiment'].rolling(window=7).mean()
        df['sentiment_volatility'] = df['Sentiment'].rolling(window=7).std()
    else:
        # Create dummy sentiment features if not available
        df['sentiment_lag_1'] = 0
        df['sentiment_ma_3'] = 0
        df['sentiment_ma_7'] = 0
        df['sentiment_volatility'] = 0
    
    # Technical indicators
    df['rsi'] = calculate_rsi(df['y'], window=14)
    df['bollinger_upper'], df['bollinger_lower'] = calculate_bollinger_bands(df['y'], window=20)
    df['bollinger_position'] = (df['y'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
    
    # Volume features (if available)
    if 'Volume' in df.columns:
        df['volume_ma_3'] = df['Volume'].rolling(window=3).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_3']
    else:
        df['volume_ma_3'] = 1
        df['volume_ratio'] = 1
    
    return df

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def train_xgboost_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict:
    """
    Train XGBoost model for stock prediction
    """
    # Prepare features
    df_features = prepare_xgboost_features(df.copy())
    
    # Select feature columns
    feature_cols = [
        'close_lag_1', 'close_lag_2', 'close_lag_3',
        'ma_3', 'ma_7', 'ma_14',
        'volatility_3', 'volatility_7',
        'price_change_1', 'price_change_3', 'price_change_7',
        'sentiment_lag_1', 'sentiment_ma_3', 'sentiment_ma_7', 'sentiment_volatility',
        'rsi', 'bollinger_position',
        'volume_ma_3', 'volume_ratio'
    ]
    
    # Remove rows with NaN values
    df_clean = df_features.dropna()
    
    if len(df_clean) < 20:
        raise ValueError("Insufficient data for training XGBoost model")
    
    # Prepare features and target
    X = df_clean[feature_cols]
    y = df_clean['y']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

def predict_xgboost(model_dict: Dict, df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Make predictions using trained XGBoost model
    """
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_cols = model_dict['feature_cols']
    
    # Prepare features for prediction
    df_features = prepare_xgboost_features(df.copy())
    
    # Get the last row for prediction
    last_row = df_features.iloc[-1:].copy()
    
    predictions = []
    current_data = last_row.copy()
    
    for i in range(periods):
        # Prepare features for this prediction
        X_pred = current_data[feature_cols].fillna(0)
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        pred = model.predict(X_pred_scaled)[0]
        predictions.append(pred)
        
        # Update data for next prediction
        new_row = current_data.copy()
        new_row['y'] = pred
        new_row['ds'] = pd.to_datetime(new_row['ds']) + pd.Timedelta(days=1)
        
        # Shift lag features
        new_row['close_lag_1'] = current_data['y'].iloc[0]
        new_row['close_lag_2'] = current_data['close_lag_1'].iloc[0]
        new_row['close_lag_3'] = current_data['close_lag_2'].iloc[0]
        
        # Update rolling features (simplified)
        new_row['ma_3'] = (current_data['y'].iloc[0] + current_data['close_lag_1'].iloc[0] + current_data['close_lag_2'].iloc[0]) / 3
        new_row['ma_7'] = current_data['ma_7'].iloc[0]  # Simplified
        new_row['ma_14'] = current_data['ma_14'].iloc[0]  # Simplified
        
        # Update other features (simplified)
        new_row['volatility_3'] = current_data['volatility_3'].iloc[0]
        new_row['volatility_7'] = current_data['volatility_7'].iloc[0]
        new_row['price_change_1'] = (pred - current_data['y'].iloc[0]) / current_data['y'].iloc[0]
        new_row['price_change_3'] = current_data['price_change_1'].iloc[0]
        new_row['price_change_7'] = current_data['price_change_3'].iloc[0]
        
        # Update sentiment features (simplified)
        new_row['sentiment_lag_1'] = current_data['Sentiment'].iloc[0] if 'Sentiment' in current_data.columns else 0
        new_row['sentiment_ma_3'] = current_data['sentiment_ma_3'].iloc[0]
        new_row['sentiment_ma_7'] = current_data['sentiment_ma_7'].iloc[0]
        new_row['sentiment_volatility'] = current_data['sentiment_volatility'].iloc[0]
        
        # Update technical indicators (simplified)
        new_row['rsi'] = current_data['rsi'].iloc[0]
        new_row['bollinger_position'] = current_data['bollinger_position'].iloc[0]
        
        # Update volume features (simplified)
        new_row['volume_ma_3'] = current_data['volume_ma_3'].iloc[0]
        new_row['volume_ratio'] = current_data['volume_ratio'].iloc[0]
        
        current_data = new_row
    
    # Create prediction DataFrame
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
    
    result_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions,
        'yhat_lower': [p * 0.95 for p in predictions],  # Simplified confidence intervals
        'yhat_upper': [p * 1.05 for p in predictions]
    })
    
    return result_df

def main():
    """Main function to test XGBoost model"""
    from modeling.prophet_model import load_features
    
    print("Loading data and training XGBoost model...")
    
    # Load data
    df = load_features()
    
    if len(df) < 50:
        print("Insufficient data for XGBoost training")
        return
    
    # Train model
    model_results = train_xgboost_model(df)
    
    print(f"XGBoost Model Results:")
    print(f"Train RMSE: {model_results['train_rmse']:.4f}")
    print(f"Test RMSE: {model_results['test_rmse']:.4f}")
    print(f"Train MAE: {model_results['train_mae']:.4f}")
    print(f"Test MAE: {model_results['test_mae']:.4f}")
    
    print(f"\nTop 10 Most Important Features:")
    print(model_results['feature_importance'].head(10))
    
    # Make predictions
    predictions = predict_xgboost(model_results, df, periods=30)
    print(f"\nPredictions for next 30 days:")
    print(predictions.head())

if __name__ == "__main__":
    main()
