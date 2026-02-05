"""
Generate Clean System Architecture Diagram for Stock Prediction System
Simplified version showing main components and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
colors = {
    'frontend': '#4A90E2',      # Blue
    'api': '#50C878',            # Green
    'data': '#FF6B6B',           # Red
    'ml': '#FFA500',             # Orange
    'feature': '#9B59B6',        # Purple
    'eval': '#1ABC9C',           # Teal
    'storage': '#95A5A6'         # Gray
}

# ==================== TITLE ====================
ax.text(8, 9.5, 'Stock Prediction System - Architecture', 
        ha='center', va='center',
        fontsize=22, fontweight='bold', color='#2C3E50')

# ==================== LAYER 1: FRONTEND ====================
frontend_box = FancyBboxPatch((1, 7.5), 3, 1.2, 
                               boxstyle="round,pad=0.15", 
                               facecolor=colors['frontend'],
                               edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(frontend_box)
ax.text(2.5, 8.3, 'Frontend', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='white')
ax.text(2.5, 7.9, 'React Dashboard', ha='center', va='center', 
        fontsize=11, color='white', style='italic')

# ==================== LAYER 2: API ====================
api_box = FancyBboxPatch((5.5, 7.5), 3, 1.2,
                         boxstyle="round,pad=0.15",
                         facecolor=colors['api'],
                         edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(api_box)
ax.text(7, 8.3, 'API Layer', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')
ax.text(7, 7.9, 'FastAPI REST', ha='center', va='center',
        fontsize=11, color='white', style='italic')

# ==================== LAYER 3: DATA INGESTION ====================
data_box = FancyBboxPatch((10, 7.5), 3, 1.2,
                          boxstyle="round,pad=0.15",
                          facecolor=colors['data'],
                          edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(data_box)
ax.text(11.5, 8.3, 'Data Ingestion', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')
ax.text(11.5, 7.9, 'yfinance + News', ha='center', va='center',
        fontsize=11, color='white', style='italic')

# ==================== LAYER 4: FEATURE ENGINEERING ====================
feature_box = FancyBboxPatch((13.5, 7.5), 2.5, 1.2,
                             boxstyle="round,pad=0.15",
                             facecolor=colors['feature'],
                             edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(feature_box)
ax.text(14.75, 8.3, 'Features', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')
ax.text(14.75, 7.9, 'RSI, MACD, MA', ha='center', va='center',
        fontsize=11, color='white', style='italic')

# ==================== ML MODELS ====================
ml_box = FancyBboxPatch((1, 5), 7, 1.5,
                        boxstyle="round,pad=0.15",
                        facecolor=colors['ml'],
                        edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(ml_box)
ax.text(4.5, 6.2, 'ML Models', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

# Prophet
prophet_circle = mpatches.Circle((2.5, 5.5), 0.4, 
                                 facecolor='white', 
                                 edgecolor='#2C3E50', linewidth=2)
ax.add_patch(prophet_circle)
ax.text(2.5, 5.5, 'P', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2C3E50')
ax.text(2.5, 5.1, 'Prophet', ha='center', va='center',
        fontsize=10, color='white', style='italic')

# XGBoost
xgboost_circle = mpatches.Circle((4.5, 5.5), 0.4,
                                 facecolor='white',
                                 edgecolor='#2C3E50', linewidth=2)
ax.add_patch(xgboost_circle)
ax.text(4.5, 5.5, 'X', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2C3E50')
ax.text(4.5, 5.1, 'XGBoost', ha='center', va='center',
        fontsize=10, color='white', style='italic')

# Signals
signals_circle = mpatches.Circle((6.5, 5.5), 0.4,
                                 facecolor='white',
                                 edgecolor='#2C3E50', linewidth=2)
ax.add_patch(signals_circle)
ax.text(6.5, 5.5, 'S', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2C3E50')
ax.text(6.5, 5.1, 'Signals', ha='center', va='center',
        fontsize=10, color='white', style='italic')

# ==================== EVALUATION ====================
eval_box = FancyBboxPatch((9, 5), 3, 1.5,
                          boxstyle="round,pad=0.15",
                          facecolor=colors['eval'],
                          edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(eval_box)
ax.text(10.5, 6.2, 'Evaluation', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')
ax.text(10.5, 5.7, 'RMSE, MAE, MAPE', ha='center', va='center',
        fontsize=10, color='white', style='italic')
ax.text(10.5, 5.3, 'Directional Accuracy', ha='center', va='center',
        fontsize=10, color='white', style='italic')

# ==================== STORAGE ====================
storage_box = FancyBboxPatch((12.5, 5), 3.5, 1.5,
                             boxstyle="round,pad=0.15",
                             facecolor=colors['storage'],
                             edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(storage_box)
ax.text(14.25, 6.2, 'Storage', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')
ax.text(14.25, 5.7, 'Model Cache', ha='center', va='center',
        fontsize=10, color='white', style='italic')
ax.text(14.25, 5.3, 'Ticker Lists', ha='center', va='center',
        fontsize=10, color='white', style='italic')

# ==================== ADVANCED FEATURES ====================
advanced_box = FancyBboxPatch((1, 2.5), 15, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#3498DB',
                              edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(advanced_box)
ax.text(8.5, 3.7, 'Advanced Features', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

features_list = [
    (2.5, 3.2, 'Backtesting'),
    (4.5, 3.2, 'Paper Trading'),
    (6.5, 3.2, 'News Analysis'),
    (8.5, 3.2, 'Risk Management'),
    (10.5, 3.2, 'Portfolio Metrics'),
    (12.5, 3.2, 'Stock Comparison'),
    (14.5, 3.2, 'Market Insights')
]
for x, y, label in features_list:
    ax.text(x, y, label, ha='center', va='center',
            fontsize=10, color='white', style='italic')

# ==================== DATA FLOW ARROWS ====================

# Frontend to API
arrow1 = FancyArrowPatch((4, 8.1), (5.5, 8.1),
                         arrowstyle='->', mutation_scale=25,
                         color='#2C3E50', linewidth=3)
ax.add_patch(arrow1)
ax.text(4.75, 8.4, 'HTTP', ha='center', va='center',
        fontsize=9, color='#2C3E50', fontweight='bold')

# API to Data
arrow2 = FancyArrowPatch((8.5, 8.1), (10, 8.1),
                         arrowstyle='->', mutation_scale=25,
                         color='#2C3E50', linewidth=3)
ax.add_patch(arrow2)
ax.text(9.25, 8.4, 'Fetch', ha='center', va='center',
        fontsize=9, color='#2C3E50', fontweight='bold')

# Data to Features
arrow3 = FancyArrowPatch((13, 8.1), (13.5, 8.1),
                         arrowstyle='->', mutation_scale=25,
                         color='#2C3E50', linewidth=3)
ax.add_patch(arrow3)
ax.text(13.25, 8.4, 'Process', ha='center', va='center',
        fontsize=9, color='#2C3E50', fontweight='bold')

# Features to ML (diagonal)
arrow4 = FancyArrowPatch((14.75, 7.5), (4.5, 6.5),
                         arrowstyle='->', mutation_scale=25,
                         color='#2C3E50', linewidth=3)
ax.add_patch(arrow4)
ax.text(9.5, 7, 'Train', ha='center', va='center',
        fontsize=9, color='#2C3E50', fontweight='bold', rotation=-20)

# ML to Evaluation
arrow5 = FancyArrowPatch((8, 5.75), (9, 5.75),
                         arrowstyle='->', mutation_scale=25,
                         color='#2C3E50', linewidth=3)
ax.add_patch(arrow5)
ax.text(8.5, 6, 'Evaluate', ha='center', va='center',
        fontsize=9, color='#2C3E50', fontweight='bold')

# ML to Storage
arrow6 = FancyArrowPatch((8, 5.5), (12.5, 5.75),
                         arrowstyle='->', mutation_scale=25,
                         color='#2C3E50', linewidth=3)
ax.add_patch(arrow6)
ax.text(10.25, 5.6, 'Cache', ha='center', va='center',
        fontsize=9, color='#2C3E50', fontweight='bold', rotation=-10)

# ML to Advanced Features
arrow7 = FancyArrowPatch((4.5, 5), (8.5, 4),
                         arrowstyle='->', mutation_scale=25,
                         color='#2C3E50', linewidth=3)
ax.add_patch(arrow7)
ax.text(6.5, 4.4, 'Features', ha='center', va='center',
        fontsize=9, color='#2C3E50', fontweight='bold', rotation=-20)

# API Response to Frontend
arrow8 = FancyArrowPatch((5.5, 7.5), (4, 8.1),
                         arrowstyle='->', mutation_scale=25,
                         color='#E74C3C', linewidth=3, linestyle='--')
ax.add_patch(arrow8)
ax.text(4.75, 7.7, 'Response', ha='center', va='center',
        fontsize=9, color='#E74C3C', fontweight='bold')

# ==================== DATA FLOW SUMMARY ====================
flow_box = FancyBboxPatch((1, 0.5), 15, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor='#ECF0F1',
                          edgecolor='#2C3E50', linewidth=2)
ax.add_patch(flow_box)
ax.text(8.5, 1.5, 'Data Flow', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2C3E50')
flow_text = 'User Request → Frontend → API → Data → Features → ML Models → Evaluation → Response'
ax.text(8.5, 0.9, flow_text, ha='center', va='center',
        fontsize=10, color='#34495E', style='italic')

# ==================== LEGEND ====================
legend_x = 0.5
legend_y = 0.3
legend_items = [
    ('Frontend', colors['frontend']),
    ('API', colors['api']),
    ('Data', colors['data']),
    ('Features', colors['feature']),
    ('ML Models', colors['ml']),
    ('Evaluation', colors['eval']),
    ('Storage', colors['storage'])
]

for i, (label, color) in enumerate(legend_items):
    x_pos = legend_x + i * 2.2
    rect = Rectangle((x_pos, legend_y), 0.3, 0.15, 
                     facecolor=color, edgecolor='#2C3E50', linewidth=1)
    ax.add_patch(rect)
    ax.text(x_pos + 0.4, legend_y + 0.075, label, ha='left', va='center',
            fontsize=9, color='#2C3E50', fontweight='bold')

# Save the diagram
plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(output_dir, 'system_architecture_diagram.png')
pdf_path = os.path.join(output_dir, 'system_architecture_diagram.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print("Clean architecture diagram generated successfully!")
print("Files created:")
print(f"   - {png_path}")
print(f"   - {pdf_path}")

# Don't show plot in non-interactive environment
# plt.show()
