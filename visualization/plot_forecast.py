import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from modeling.prophet_model import load_features, train_prophet

def plot_forecast_with_sentiment(df, forecast):
    """Enhanced forecast plot with sentiment overlays"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Use close_price if available (for return-based data), otherwise use y
    price_col = 'close_price' if 'close_price' in df.columns else 'y'
    
    # Main price plot
    ax1.plot(df['ds'], df[price_col], label='Actual Price', color='black', linewidth=2)
    ax1.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue', linewidth=2)
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='lightblue', alpha=0.3, label='Confidence Interval')
    
    # Sentiment overlay
    ax1_twin = ax1.twinx()
    sentiment_colors = ['red' if x < 0 else 'green' for x in df['Sentiment']]
    ax1_twin.scatter(df['ds'], df[price_col], c=sentiment_colors, alpha=0.6, s=30, label='Sentiment')
    ax1_twin.set_ylabel('Price (Sentiment Colored)', fontsize=10)
    
    # Sentiment line plot
    ax2.plot(df['ds'], df['Sentiment'], color='purple', linewidth=1.5, label='Sentiment Score')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(df['ds'], df['Sentiment'], 0, alpha=0.3, color='purple')
    ax2.set_ylabel('Sentiment Score', fontsize=10)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Styling
    ax1.set_title('Stock Price Forecast with Sentiment Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_volatility_analysis(df, forecast):
    """Plot rolling volatility vs forecast confidence intervals"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Use close_price if available (for return-based data), otherwise use y
    price_col = 'close_price' if 'close_price' in df.columns else 'y'
    
    # Price and confidence intervals
    ax1.plot(df['ds'], df[price_col], label='Actual Price', color='black', linewidth=2)
    ax1.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue', linewidth=2)
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='lightblue', alpha=0.3, label='Confidence Interval')
    
    # Volatility overlay
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['ds'], df['Volatility'], color='red', linewidth=1.5, label='Rolling Volatility')
    ax1_twin.set_ylabel('Volatility', color='red', fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Volatility vs Confidence Width comparison
    confidence_width = forecast['yhat_upper'] - forecast['yhat_lower']
    ax2.plot(df['ds'], df['Volatility'], label='Historical Volatility', color='red', linewidth=2)
    ax2.plot(forecast['ds'], confidence_width, label='Forecast Confidence Width', color='blue', linewidth=2)
    ax2.set_ylabel('Volatility / Confidence Width', fontsize=10)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax1.set_title('Volatility Analysis vs Forecast Confidence', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_interactive_dashboard(df, forecast):
    """Create interactive Plotly dashboard"""
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Stock Price Forecast', 'Sentiment Analysis', 'Volatility Analysis'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Use close_price if available (for return-based data), otherwise use y
    price_col = 'close_price' if 'close_price' in df.columns else 'y'
    
    # Price forecast
    fig.add_trace(go.Scatter(x=df['ds'], y=df[price_col], name='Actual Price', 
                            line=dict(color='black', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', 
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                            fill=None, mode='lines', line_color='rgba(0,0,0,0)', 
                            showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                            name='Confidence Interval'), row=1, col=1)
    
    # Sentiment analysis
    fig.add_trace(go.Scatter(x=df['ds'], y=df['Sentiment'], name='Sentiment Score', 
                            line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    
    # Volatility analysis
    fig.add_trace(go.Scatter(x=df['ds'], y=df['Volatility'], name='Volatility', 
                            line=dict(color='red', width=2)), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title='Interactive Stock Analysis Dashboard',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
    fig.update_yaxes(title_text="Volatility", row=3, col=1)
    
    return fig

def export_plots(df, forecast, output_dir='output'):
    """Export all plots to PNG files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export matplotlib plots
    fig1 = plot_forecast_with_sentiment(df, forecast)
    fig1.savefig(f'{output_dir}/forecast_with_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = plot_volatility_analysis(df, forecast)
    fig2.savefig(f'{output_dir}/volatility_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Export interactive dashboard
    interactive_fig = create_interactive_dashboard(df, forecast)
    interactive_fig.write_html(f'{output_dir}/interactive_dashboard.html')
    
    print(f"Plots exported to {output_dir}/ directory:")
    print("- forecast_with_sentiment.png")
    print("- volatility_analysis.png") 
    print("- interactive_dashboard.html")

def main():
    print("Loading data and training model...")
    df = load_features(ticker="AAPL", use_real_sentiment=True)
    forecast = train_prophet(df, periods=30)
    
    print("Creating enhanced visualizations...")
    
    # Display plots
    plot_forecast_with_sentiment(df, forecast)
    plt.show()
    
    plot_volatility_analysis(df, forecast)
    plt.show()
    
    # Export all plots
    export_plots(df, forecast)
    
    print("Creating interactive dashboard...")
    interactive_fig = create_interactive_dashboard(df, forecast)
    interactive_fig.show()

if __name__ == "__main__":
    main()
