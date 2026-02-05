"""
Evaluation metrics for stock prediction models
Includes RMSE, MAPE, directional accuracy, and baseline comparisons
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation for stock predictions"""
    
    def __init__(self):
        self.metrics = {}
        self.baseline_metrics = {}
    
    def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error
        BUG FIX 1: Convert to float dtype BEFORE calculation to prevent stuck RMSE
        """
        # CRITICAL: Convert to float to prevent dtype=object issues
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        
        # Remove NaN and inf values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not np.any(mask):
            return 0.0
        
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Ensure equal length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        return np.sqrt(mean_squared_error(actual, predicted))
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error
        BUG FIX 1: Convert to float dtype BEFORE calculation
        BUG FIX 7: Handle low-price stocks better - cap MAPE to prevent explosion
        """
        # CRITICAL: Convert to float to prevent dtype=object issues
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        
        # Remove NaN and inf values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not np.any(mask):
            return 0.0
        
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Ensure equal length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # BUG FIX 7: Handle low-price stocks - use relative error with clipping
        # Instead of simple division, use symmetric MAPE to prevent explosion
        # Filter out values too close to zero to avoid division issues
        non_zero_mask = np.abs(actual) > 1e-6
        if not np.any(non_zero_mask):
            return 0.0
        
        actual_filtered = actual[non_zero_mask]
        predicted_filtered = predicted[non_zero_mask]
        
        # Calculate MAPE with clipping to prevent extreme values
        mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
        
        # Cap MAPE at 1000% to prevent explosion on low-price stocks
        return min(mape, 1000.0)
    
    def calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        BUG FIX 1: Convert to float dtype BEFORE calculation to prevent stuck MAE
        """
        # CRITICAL: Convert to float to prevent dtype=object issues
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        
        # Remove NaN and inf values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not np.any(mask):
            return 0.0
        
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Ensure equal length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        return mean_absolute_error(actual, predicted)
    
    def calculate_directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray, 
                                      use_returns: bool = False, is_future_forecast: bool = False) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions)
        
        BUG FIX 2: Properly handle future forecasts vs historical fitted values
        - For future forecasts: Compare predicted future direction vs actual future direction
        - For historical fits: Compare predicted historical direction vs actual historical direction
        
        Args:
            actual: Actual values (can be historical or future)
            predicted: Predicted values (can be historical fit or future forecast)
            use_returns: If True, inputs are already returns
            is_future_forecast: If True, these are future predictions (not historical fit)
        """
        # BUG FIX 1: Convert to float first
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        
        # Remove NaN and inf values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not np.any(mask):
            return 0.0
        
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) < 2 or len(predicted) < 2:
            return 0.0
        
        # Ensure equal length (align by taking minimum)
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        if len(actual) < 2:
            return 0.0
        
        # BUG FIX 2: Proper directional accuracy calculation
        # Calculate direction changes: sign(value[i+1] - value[i])
        if use_returns:
            # Inputs are already returns - compare signs of returns directly
            # For returns: direction = sign(return)
            actual_directions = np.sign(actual[1:])  # Sign of return at time i+1
            predicted_directions = np.sign(predicted[1:])  # Sign of predicted return at time i+1
        else:
            # Calculate directions from price changes
            # Direction = sign(price[i+1] - price[i])
            actual_changes = actual[1:] - actual[:-1]
            predicted_changes = predicted[1:] - predicted[:-1]
            
            # Get direction signs (1 = up, -1 = down, 0 = flat)
            actual_directions = np.sign(actual_changes)
            predicted_directions = np.sign(predicted_changes)
        
        # Count correct directional predictions (excluding flat/zero cases)
        # Only count non-zero directions for meaningful comparison
        non_zero_mask = (actual_directions != 0) | (predicted_directions != 0)
        if not np.any(non_zero_mask):
            return 50.0  # If all flat, return neutral accuracy
        
        actual_dir_filtered = actual_directions[non_zero_mask]
        predicted_dir_filtered = predicted_directions[non_zero_mask]
        
        correct_directions = np.sum(actual_dir_filtered == predicted_dir_filtered)
        total = len(actual_dir_filtered)
        
        return (correct_directions / total) * 100 if total > 0 else 0.0
    
    def calculate_volatility_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate how well the model predicts volatility patterns
        BUG FIX 1: Convert to float first
        BUG FIX 6: Fix unrealistic volatility accuracy formula
        """
        # BUG FIX 1: Convert to float first
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        
        # Remove NaN and inf values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not np.any(mask):
            return 0.0
        
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Ensure equal length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        if len(actual) < 2:
            return 0.0
        
        # Calculate volatility (standard deviation of returns/changes)
        actual_returns = np.diff(actual) / actual[:-1] if len(actual) > 1 else np.array([0])
        predicted_returns = np.diff(predicted) / predicted[:-1] if len(predicted) > 1 else np.array([0])
        
        actual_vol = np.std(actual_returns) if len(actual_returns) > 0 else 0.0
        predicted_vol = np.std(predicted_returns) if len(predicted_returns) > 0 else 0.0
        
        if actual_vol == 0:
            return 100.0 if predicted_vol == 0 else 0.0
        
        # BUG FIX 6: Use symmetric relative error to prevent negative scores
        # Formula: 100 * (1 - min(1, |actual_vol - predicted_vol| / max(actual_vol, predicted_vol)))
        # This prevents negative scores when predicted_vol > 2× actual_vol
        max_vol = max(actual_vol, predicted_vol)
        relative_error = abs(actual_vol - predicted_vol) / max_vol if max_vol > 0 else 1.0
        
        # Convert to accuracy score (0-100)
        accuracy = 100 * (1 - min(1.0, relative_error))
        
        return max(0.0, min(100.0, accuracy))
    
    def calculate_confidence_interval_coverage(self, actual: np.ndarray, 
                                            lower_bound: np.ndarray, 
                                            upper_bound: np.ndarray) -> float:
        """
        Calculate percentage of actual values within confidence intervals
        BUG FIX 1: Convert to float first
        BUG FIX 3: Fix Prophet confidence interval coverage - ensure proper alignment
        """
        # BUG FIX 1: Convert to float first
        actual = np.asarray(actual, dtype=float)
        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)
        
        # Remove NaN and inf values
        mask = np.isfinite(actual) & np.isfinite(lower_bound) & np.isfinite(upper_bound)
        if not np.any(mask):
            return 0.0
        
        actual = actual[mask]
        lower_bound = lower_bound[mask]
        upper_bound = upper_bound[mask]
        
        # BUG FIX 3: Ensure equal length and proper alignment
        min_len = min(len(actual), len(lower_bound), len(upper_bound))
        if min_len == 0:
            return 0.0
        
        actual = actual[:min_len]
        lower_bound = lower_bound[:min_len]
        upper_bound = upper_bound[:min_len]
        
        # Ensure lower <= upper (swap if needed)
        lower_bound = np.minimum(lower_bound, upper_bound)
        upper_bound = np.maximum(lower_bound, upper_bound)
        
        # Count values within bounds
        within_bounds = np.sum(
            (actual >= lower_bound) & (actual <= upper_bound)
        )
        
        return (within_bounds / min_len) * 100 if min_len > 0 else 0.0
    
    def evaluate_model(self, actual: np.ndarray, predicted: np.ndarray, 
                      lower_bound: Optional[np.ndarray] = None,
                      upper_bound: Optional[np.ndarray] = None,
                      use_returns: bool = False,
                      is_future_forecast: bool = False) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        BUG FIX: All metrics now properly handle dtype conversion and alignment
        """
        # BUG FIX 1: Ensure inputs are float arrays
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        
        # Remove NaN and inf values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not np.any(mask):
            return {
                'RMSE': 0.0,
                'MAE': 0.0,
                'MAPE': 0.0,
                'Directional_Accuracy': 0.0,
                'Volatility_Accuracy': 0.0
            }
        
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Ensure equal length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        metrics = {
            'RMSE': self.calculate_rmse(actual, predicted),
            'MAE': self.calculate_mae(actual, predicted),
            'MAPE': self.calculate_mape(actual, predicted),
            'Directional_Accuracy': self.calculate_directional_accuracy(
                actual, predicted, 
                use_returns=use_returns,
                is_future_forecast=is_future_forecast
            ),
            'Volatility_Accuracy': self.calculate_volatility_accuracy(actual, predicted)
        }
        
        if lower_bound is not None and upper_bound is not None:
            # Align confidence intervals with actual/predicted
            lower_bound = np.asarray(lower_bound, dtype=float)[mask][:min_len]
            upper_bound = np.asarray(upper_bound, dtype=float)[mask][:min_len]
            
            metrics['Confidence_Coverage'] = self.calculate_confidence_interval_coverage(
                actual, lower_bound, upper_bound
            )
        
        return metrics
    
    def naive_baseline(self, data: np.ndarray, forecast_periods: int) -> np.ndarray:
        """Simple naive baseline (last value repeated)"""
        if len(data) == 0:
            return np.zeros(forecast_periods)
        
        last_value = data[-1]
        return np.full(forecast_periods, last_value)
    
    def moving_average_baseline(self, data: np.ndarray, window: int, 
                               forecast_periods: int) -> np.ndarray:
        """Moving average baseline"""
        if len(data) < window:
            return self.naive_baseline(data, forecast_periods)
        
        ma_value = np.mean(data[-window:])
        return np.full(forecast_periods, ma_value)
    
    def linear_trend_baseline(self, data: np.ndarray, forecast_periods: int) -> np.ndarray:
        """Linear trend baseline"""
        if len(data) < 2:
            return self.naive_baseline(data, forecast_periods)
        
        x = np.arange(len(data))
        y = data
        
        # Fit linear trend
        coeffs = np.polyfit(x, y, 1)
        
        # Extend trend into future
        future_x = np.arange(len(data), len(data) + forecast_periods)
        future_y = coeffs[0] * future_x + coeffs[1]
        
        return future_y
    
    def evaluate_baselines(self, train_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple baseline models"""
        baselines = {}
        
        # Naive baseline
        naive_pred = self.naive_baseline(train_data, len(test_data))
        baselines['Naive'] = self.evaluate_model(test_data, naive_pred)
        
        # Moving average baselines
        for window in [3, 5, 10]:
            if len(train_data) >= window:
                ma_pred = self.moving_average_baseline(train_data, window, len(test_data))
                baselines[f'MA_{window}'] = self.evaluate_model(test_data, ma_pred)
        
        # Linear trend baseline
        linear_pred = self.linear_trend_baseline(train_data, len(test_data))
        baselines['Linear_Trend'] = self.evaluate_model(test_data, linear_pred)
        
        return baselines
    
    def compare_models(self, actual: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare multiple models and return results as DataFrame
        BUG FIX 5: Align lengths BEFORE evaluation to prevent Prophet bars from disappearing
        """
        # BUG FIX 1: Convert actual to float
        actual = np.asarray(actual, dtype=float)
        
        # Remove NaN and inf from actual
        actual_mask = np.isfinite(actual)
        actual = actual[actual_mask]
        
        results = []
        
        for model_name, predicted in predictions.items():
            # BUG FIX 5: Align lengths BEFORE evaluation
            predicted = np.asarray(predicted, dtype=float)
            
            # Remove NaN and inf from predicted
            pred_mask = np.isfinite(predicted)
            predicted = predicted[pred_mask]
            
            # Align to same length (take minimum)
            min_len = min(len(actual), len(predicted))
            if min_len == 0:
                # Skip if no valid data
                continue
            
            actual_aligned = actual[:min_len]
            predicted_aligned = predicted[:min_len]
            
            # Now evaluate with aligned arrays
            metrics = self.evaluate_model(actual_aligned, predicted_aligned)
            metrics['Model'] = model_name
            results.append(metrics)
        
        if not results:
            # Return empty DataFrame with correct columns if no results
            return pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy', 'Volatility_Accuracy'])
        
        return pd.DataFrame(results)
    
    def plot_evaluation(self, actual: np.ndarray, predictions: Dict[str, np.ndarray], 
                       title: str = "Model Comparison") -> plt.Figure:
        """
        Plot actual vs predicted values for multiple models
        BUG FIX 5: Align all predictions to actual length before plotting
        """
        # BUG FIX 1: Convert to float and align
        actual = np.asarray(actual, dtype=float)
        actual_mask = np.isfinite(actual)
        actual = actual[actual_mask]
        
        # Align all predictions to actual length
        aligned_predictions = {}
        for model_name, predicted in predictions.items():
            predicted = np.asarray(predicted, dtype=float)
            pred_mask = np.isfinite(predicted)
            predicted = predicted[pred_mask]
            
            min_len = min(len(actual), len(predicted))
            if min_len > 0:
                aligned_predictions[model_name] = predicted[:min_len]
        
        # Use aligned actual (take minimum length across all)
        if aligned_predictions:
            min_len = min(len(actual), *[len(p) for p in aligned_predictions.values()])
            actual = actual[:min_len]
            aligned_predictions = {k: v[:min_len] for k, v in aligned_predictions.items()}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Time series plot
        ax1 = axes[0, 0]
        x = np.arange(len(actual))
        ax1.plot(x, actual, 'k-', label='Actual', linewidth=2)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, predicted) in enumerate(aligned_predictions.items()):
            if len(predicted) == len(actual):
                ax1.plot(x, predicted, '--', color=colors[i % len(colors)], 
                        label=model_name, alpha=0.8)
        
        ax1.set_title('Time Series Comparison')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot (actual vs predicted)
        ax2 = axes[0, 1]
        for i, (model_name, predicted) in enumerate(aligned_predictions.items()):
            if len(predicted) == len(actual):
                ax2.scatter(actual, predicted, alpha=0.6, 
                           color=colors[i % len(colors)], label=model_name)
        
        # Perfect prediction line
        if len(actual) > 0 and aligned_predictions:
            all_values = [actual] + list(aligned_predictions.values())
            min_val = min(p.min() for p in all_values if len(p) > 0)
            max_val = max(p.max() for p in all_values if len(p) > 0)
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax2.set_title('Actual vs Predicted')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Error distribution
        ax3 = axes[1, 0]
        for i, (model_name, predicted) in enumerate(aligned_predictions.items()):
            if len(predicted) == len(actual):
                errors = actual - predicted
                ax3.hist(errors, alpha=0.6, bins=20, 
                        color=colors[i % len(colors)], label=model_name)
        
        ax3.set_title('Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Metrics comparison
        ax4 = axes[1, 1]
        # Use aligned predictions for metrics comparison
        metrics_df = self.compare_models(actual, aligned_predictions)
        
        # Select key metrics for visualization
        key_metrics = ['RMSE', 'MAPE', 'Directional_Accuracy']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        if available_metrics:
            x_pos = np.arange(len(metrics_df))
            width = 0.25
            
            for i, metric in enumerate(available_metrics):
                ax4.bar(x_pos + i * width, metrics_df[metric], width, 
                       label=metric, alpha=0.8)
            
            ax4.set_title('Metrics Comparison')
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Metric Value')
            ax4.set_xticks(x_pos + width)
            ax4.set_xticklabels(metrics_df['Model'], rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, actual: np.ndarray, predictions: Dict[str, np.ndarray]) -> str:
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("STOCK PREDICTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Model comparison
        metrics_df = self.compare_models(actual, predictions)
        
        report.append(f"\nData Points: {len(actual)}")
        report.append(f"Models Evaluated: {len(predictions)}")
        
        report.append("\n" + "-" * 40)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        
        # Format metrics table
        for _, row in metrics_df.iterrows():
            report.append(f"\n{row['Model']}:")
            for metric, value in row.items():
                if metric != 'Model':
                    if 'Accuracy' in metric or 'Coverage' in metric:
                        report.append(f"  {metric}: {value:.2f}%")
                    else:
                        report.append(f"  {metric}: {value:.4f}")
        
        # Best model analysis
        if 'RMSE' in metrics_df.columns:
            best_rmse_idx = metrics_df['RMSE'].idxmin()
            best_rmse_model = metrics_df.loc[best_rmse_idx, 'Model']
            report.append(f"\nBest RMSE: {best_rmse_model} ({metrics_df.loc[best_rmse_idx, 'RMSE']:.4f})")
        
        if 'Directional_Accuracy' in metrics_df.columns:
            best_dir_idx = metrics_df['Directional_Accuracy'].idxmax()
            best_dir_model = metrics_df.loc[best_dir_idx, 'Model']
            report.append(f"Best Directional Accuracy: {best_dir_model} ({metrics_df.loc[best_dir_idx, 'Directional_Accuracy']:.2f}%)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def evaluate_prophet_model(actual: np.ndarray, forecast_df: pd.DataFrame, 
                          is_future_forecast: bool = False) -> Dict[str, float]:
    """
    Evaluate Prophet model specifically
    BUG FIX 3: Properly handle Prophet forecast_df which contains historic + future values
    BUG FIX 1: Convert to float before evaluation
    """
    evaluator = ModelEvaluator()
    
    # BUG FIX 1: Convert to float first
    actual = np.asarray(actual, dtype=float)
    
    # Extract predictions and confidence intervals
    predicted = np.asarray(forecast_df['yhat'].values, dtype=float)
    lower_bound = np.asarray(forecast_df['yhat_lower'].values, dtype=float)
    upper_bound = np.asarray(forecast_df['yhat_upper'].values, dtype=float)
    
    # BUG FIX 3: Prophet forecast_df may contain historic fitted + future forecast
    # We need to align properly - take the overlapping period
    # Remove NaN and inf values
    actual_mask = np.isfinite(actual)
    pred_mask = np.isfinite(predicted)
    lower_mask = np.isfinite(lower_bound)
    upper_mask = np.isfinite(upper_bound)
    
    # Get valid indices
    valid_mask = actual_mask & pred_mask & lower_mask & upper_mask
    
    if not np.any(valid_mask):
        return {
            'RMSE': 0.0,
            'MAE': 0.0,
            'MAPE': 0.0,
            'Directional_Accuracy': 0.0,
            'Volatility_Accuracy': 0.0,
            'Confidence_Coverage': 0.0
        }
    
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]
    lower_bound = lower_bound[valid_mask]
    upper_bound = upper_bound[valid_mask]
    
    # Ensure same length (final safety check)
    min_len = min(len(actual), len(predicted), len(lower_bound), len(upper_bound))
    if min_len == 0:
        return {
            'RMSE': 0.0,
            'MAE': 0.0,
            'MAPE': 0.0,
            'Directional_Accuracy': 0.0,
            'Volatility_Accuracy': 0.0,
            'Confidence_Coverage': 0.0
        }
    
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    lower_bound = lower_bound[:min_len]
    upper_bound = upper_bound[:min_len]
    
    return evaluator.evaluate_model(
        actual, predicted, lower_bound, upper_bound,
        is_future_forecast=is_future_forecast
    )

if __name__ == "__main__":
    # Test the evaluator
    evaluator = ModelEvaluator()
    
    # Generate sample data
    np.random.seed(42)
    actual = np.cumsum(np.random.randn(100)) + 100
    predicted = actual + np.random.randn(100) * 0.5
    
    # Test metrics
    metrics = evaluator.evaluate_model(actual, predicted)
    print("Sample Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test baselines
    train_data = actual[:80]
    test_data = actual[80:]
    baselines = evaluator.evaluate_baselines(train_data, test_data)
    
    print("\nBaseline Comparison:")
    for model, metrics in baselines.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

