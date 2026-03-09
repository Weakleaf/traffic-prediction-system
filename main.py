"""
Main Application - Traffic Congestion Prediction System
CYBER TITANS - Kisii University 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from config import Config
from model import TrafficPredictionModel
from database import TrafficDatabase
from data_generator import TrafficDataGenerator

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def print_header():
    """Print system header"""
    print("="*70)
    print(" " * 15 + "TRAFFIC CONGESTION PREDICTION SYSTEM")
    print(" " * 20 + "Using CNN-LSTM Architecture")
    print("="*70)
    print("Authors: CYBER TITANS")
    print("Institution: Kisii University")
    print("Year: 2026")
    print("="*70)
    print()


def visualize_data(df):
    """Visualize traffic patterns"""
    print("\nGenerating data visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Traffic over time
    axes[0, 0].plot(df['timestamp'][:168], df['vehicle_count'][:168], linewidth=2)
    axes[0, 0].set_title('Traffic Pattern (One Week)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Vehicle Count')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Hourly average
    hourly_avg = df.groupby('hour')['vehicle_count'].mean()
    axes[0, 1].bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Average Traffic by Hour', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Average Vehicle Count')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Day of week pattern
    dow_avg = df.groupby('day_of_week')['vehicle_count'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), dow_avg.values, color='coral', alpha=0.7)
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(days)
    axes[1, 0].set_title('Average Traffic by Day of Week', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Average Vehicle Count')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Congestion distribution
    if 'congestion_level' in df.columns:
        congestion_counts = df['congestion_level'].value_counts()
        axes[1, 1].pie(congestion_counts.values, labels=congestion_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Congestion Level Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/traffic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to plots/traffic_analysis.png")


def main():
    """Main execution function"""
    print_header()
    
    # Initialize components
    config = Config()
    model = TrafficPredictionModel(config)
    db = TrafficDatabase()
    data_gen = TrafficDataGenerator()
    
    print("Step 1: Data Generation")
    print("-" * 70)
    
    # Generate or load data
    df = data_gen.load_or_generate_data(
        filepath='data/traffic_data.csv',
        n_days=365,
        force_generate=False
    )
    
    print(f"\nDataset Info:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Features: {list(df.columns)}")
    
    # Visualize data
    visualize_data(df)
    
    print("\n" + "="*70)
    print("Step 2: Data Preprocessing")
    print("-" * 70)
    
    # Prepare data
    scaled_data, features = model.prepare_data(df)
    X, y = model.create_sequences(scaled_data)
    
    print(f"\nSequence Information:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Features used: {features}")
    
    print("\n" + "="*70)
    print("Step 3: Train-Test Split")
    print("-" * 70)
    
    # Split data (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )
    
    print(f"\nData Split:")
    print(f"  Training samples:   {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation samples: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test samples:       {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("Step 4: Model Training")
    print("-" * 70)
    
    # Train model
    model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    
    # Plot training history
    model.plot_training_history(save=True)
    
    print("\n" + "="*70)
    print("Step 5: Model Evaluation")
    print("-" * 70)
    
    # Evaluate model
    predictions, metrics = model.evaluate(X_test, y_test)
    
    # Visualize predictions
    model.plot_predictions(y_test, predictions, n_samples=200, save=True)
    
    print("\n" + "="*70)
    print("Step 6: Model Persistence")
    print("-" * 70)
    
    # Save model
    model.save_model()
    
    print("\n" + "="*70)
    print("Step 7: Future Prediction Demo")
    print("-" * 70)
    
    # Demonstrate future prediction
    recent_data = df.tail(config.SEQUENCE_LENGTH * 2)
    future_predictions = model.predict_future(recent_data, hours_ahead=12)
    
    print("\nNext 12 Hours Forecast:")
    print("-" * 50)
    current_time = datetime.now()
    
    for i, pred in enumerate(future_predictions):
        target_time = current_time + pd.Timedelta(hours=i+1)
        congestion = determine_congestion(pred)
        print(f"  Hour {i+1:2d} ({target_time.strftime('%H:%M')}): "
              f"{pred:5.1f} vehicles - {congestion}")
    
    # Visualize forecast
    plt.figure(figsize=(14, 6))
    hours = range(1, len(future_predictions) + 1)
    plt.plot(hours, future_predictions, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=40, color='green', linestyle='--', label='Low Threshold', alpha=0.7)
    plt.axhline(y=70, color='orange', linestyle='--', label='High Threshold', alpha=0.7)
    plt.axhline(y=90, color='red', linestyle='--', label='Severe Threshold', alpha=0.7)
    plt.title('12-Hour Traffic Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Predicted Vehicle Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("SYSTEM IMPLEMENTATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nFinal Performance Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2_score']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"\nModel Version: {model.model_version}")
    print("\nAll outputs saved to:")
    print(f"  - Models: {config.MODEL_DIR}")
    print(f"  - Plots: {config.PLOTS_DIR}")
    print(f"  - Database: {config.DATABASE_PATH}")
    print("\nTo start the API server, run: python api.py")
    print("="*70)


def determine_congestion(traffic_value):
    """Determine congestion level"""
    if traffic_value < 40:
        return "LOW"
    elif traffic_value < 70:
        return "MODERATE"
    elif traffic_value < 90:
        return "HIGH"
    else:
        return "SEVERE"


if __name__ == "__main__":
    main()
