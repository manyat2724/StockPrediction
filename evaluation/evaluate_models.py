"""
Comprehensive model evaluation script
Compares Prophet model against various baselines
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from evaluation.metrics import ModelEvaluator, evaluate_prophet_model
from modeling.prophet_model import load_features, train_prophet
from data_ingestion.stock_fetch import fetch_stock_data
from feature_engineering.feature import simulate_sentiment_data, add_rolling_features

def split_data(df, train_ratio=0.8):
    """
    Split data into train and test sets - TIME-ORDERED (no shuffle)
    train = first 80% chronologically
    test = last 20%
    """
    # Ensure data is sorted by date (chronological order)
    if 'ds' in df.columns:
        df = df.sort_values('ds').reset_index(drop=True)
    elif 'Datetime' in df.columns:
        df = df.sort_values('Datetime').reset_index(drop=True)
    
    # Time-ordered split: first 80% for training, last 20% for testing
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

def evaluate_complete_pipeline(ticker="AAPL", use_real_sentiment=True):
    """Evaluate the complete stock prediction pipeline"""
    print("=" * 60)
    print("COMPREHENSIVE STOCK PREDICTION EVALUATION")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    df = load_features()
    print(f"   Data shape: {df.shape}")
    
    # Split data
    train_df, test_df = split_data(df, train_ratio=0.8)
    print(f"   Train set: {len(train_df)} points")
    print(f"   Test set: {len(test_df)} points")
    
    # Prepare actual values
    actual_values = test_df['y'].values
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # 1. Train Prophet model on training data
    print("\n2. Training Prophet model...")
    try:
        prophet_model = train_prophet(train_df)
        
        # Get Prophet predictions for test period
        prophet_predictions = prophet_model['yhat'].values
        prophet_lower = prophet_model['yhat_lower'].values
        prophet_upper = prophet_model['yhat_upper'].values
        
        # Align with test data
        min_len = min(len(actual_values), len(prophet_predictions))
        actual_aligned = actual_values[:min_len]
        prophet_pred_aligned = prophet_predictions[:min_len]
        prophet_lower_aligned = prophet_lower[:min_len]
        prophet_upper_aligned = prophet_upper[:min_len]
        
        prophet_metrics = evaluator.evaluate_model(
            actual_aligned, prophet_pred_aligned, 
            prophet_lower_aligned, prophet_upper_aligned
        )
        print("   Prophet model trained successfully")
        
    except Exception as e:
        print(f"   Prophet training failed: {e}")
        prophet_metrics = {}
        prophet_pred_aligned = np.zeros_like(actual_values)
    
    # 2. Evaluate baselines
    print("\n3. Evaluating baseline models...")
    train_values = train_df['y'].values
    baselines = evaluator.evaluate_baselines(train_values, actual_values)
    
    # 3. Prepare predictions dictionary
    predictions = {
        'Prophet': prophet_pred_aligned
    }
    
    # Add baseline predictions
    for model_name, metrics in baselines.items():
        if 'Naive' in model_name:
            pred = evaluator.naive_baseline(train_values, len(actual_values))
        elif 'MA_' in model_name:
            window = int(model_name.split('_')[1])
            pred = evaluator.moving_average_baseline(train_values, window, len(actual_values))
        elif 'Linear' in model_name:
            pred = evaluator.linear_trend_baseline(train_values, len(actual_values))
        else:
            continue
        
        predictions[model_name] = pred
    
    # 4. Generate comprehensive comparison
    print("\n4. Generating comprehensive evaluation...")
    
    # Create comparison DataFrame
    all_metrics = {}
    all_metrics['Prophet'] = prophet_metrics
    
    for model_name, metrics in baselines.items():
        all_metrics[model_name] = metrics
    
    # Convert to DataFrame for easy comparison
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df = metrics_df.fillna(0)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Display results
    print("\nPerformance Metrics:")
    print("-" * 40)
    for model in metrics_df.index:
        print(f"\n{model}:")
        for metric in ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']:
            if metric in metrics_df.columns:
                value = metrics_df.loc[model, metric]
                if 'Accuracy' in metric:
                    print(f"  {metric}: {value:.2f}%")
                else:
                    print(f"  {metric}: {value:.4f}")
    
    # Find best models
    print("\n" + "-" * 40)
    print("BEST PERFORMING MODELS:")
    print("-" * 40)
    
    if 'RMSE' in metrics_df.columns:
        best_rmse = metrics_df['RMSE'].idxmin()
        print(f"Lowest RMSE: {best_rmse} ({metrics_df.loc[best_rmse, 'RMSE']:.4f})")
    
    if 'Directional_Accuracy' in metrics_df.columns:
        best_dir = metrics_df['Directional_Accuracy'].idxmax()
        print(f"Best Directional Accuracy: {best_dir} ({metrics_df.loc[best_dir, 'Directional_Accuracy']:.2f}%)")
    
    if 'MAPE' in metrics_df.columns:
        best_mape = metrics_df['MAPE'].idxmin()
        print(f"Lowest MAPE: {best_mape} ({metrics_df.loc[best_mape, 'MAPE']:.2f}%)")
    
    # 5. Generate visualization
    print("\n5. Generating evaluation plots...")
    try:
        fig = evaluator.plot_evaluation(actual_aligned, predictions, 
                                      f"Model Evaluation - {ticker}")
        
        # Save plot
        os.makedirs('output', exist_ok=True)
        fig.savefig('output/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("   Evaluation plot saved: output/model_evaluation.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"   Plot generation failed: {e}")
    
    # 6. Generate detailed report
    print("\n6. Generating detailed report...")
    report = evaluator.generate_report(actual_aligned, predictions)
    
    # Save report
    with open('output/evaluation_report.txt', 'w') as f:
        f.write(report)
    print("   Report saved: output/evaluation_report.txt")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Test Period: {len(actual_values)} data points")
    print(f"Models Evaluated: {len(predictions)}")
    print(f"Real Sentiment Used: {use_real_sentiment}")
    
    if prophet_metrics:
        print(f"\nProphet Model Performance:")
        print(f"  RMSE: {prophet_metrics.get('RMSE', 'N/A'):.4f}")
        print(f"  Directional Accuracy: {prophet_metrics.get('Directional_Accuracy', 'N/A'):.2f}%")
        print(f"  MAPE: {prophet_metrics.get('MAPE', 'N/A'):.2f}%")
    
    print("\nFiles generated:")
    print("  - output/model_evaluation.png")
    print("  - output/evaluation_report.txt")
    
    return metrics_df, predictions, actual_aligned

def main():
    """Main evaluation function"""
    print("Starting comprehensive model evaluation...")
    
    # Run evaluation
    metrics_df, predictions, actual = evaluate_complete_pipeline(
        ticker="AAPL", 
        use_real_sentiment=True
    )
    
    print("\nEvaluation completed successfully!")
    return metrics_df, predictions, actual

if __name__ == "__main__":
    main()

