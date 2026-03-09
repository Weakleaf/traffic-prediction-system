"""
Quick Start Script - Traffic Prediction System
Run this to set up and test the system quickly
"""

import os
import sys

def print_banner():
    print("="*70)
    print(" " * 15 + "TRAFFIC PREDICTION SYSTEM")
    print(" " * 20 + "Quick Start Setup")
    print("="*70)
    print("CYBER TITANS - Kisii University 2026")
    print("="*70)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required = ['numpy', 'pandas', 'tensorflow', 'sklearn', 'flask', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    dirs = ['data', 'models', 'logs', 'plots']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}/")
    
    print("\n✓ Directories created!")

def run_demo():
    """Run a quick demo"""
    print("\n" + "="*70)
    print("Running Quick Demo...")
    print("="*70)
    
    try:
        from config import Config
        from model import TrafficPredictionModel
        from data_generator import TrafficDataGenerator
        from datetime import datetime
        import numpy as np
        
        # Initialize
        config = Config()
        model = TrafficPredictionModel(config)
        data_gen = TrafficDataGenerator()
        
        # Generate small dataset
        print("\n1. Generating sample data (30 days)...")
        df = data_gen.generate_traffic_data(
            datetime(2024, 1, 1), 
            n_days=30, 
            save_to_db=False
        )
        print(f"   Generated {len(df)} records")
        
        # Prepare data
        print("\n2. Preparing data...")
        scaled_data, features = model.prepare_data(df)
        X, y = model.create_sequences(scaled_data)
        print(f"   Created {len(X)} sequences")
        
        # Split data
        print("\n3. Splitting data...")
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False
        )
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train model (quick)
        print("\n4. Training model (10 epochs - quick demo)...")
        model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        
        # Evaluate
        print("\n5. Evaluating model...")
        predictions, metrics = model.evaluate(X_test, y_test)
        
        print("\n" + "="*70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nPerformance Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2_score']:.4f}")
        
        # Save model
        print("\n6. Saving model...")
        model.save_model()
        
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("1. Run full training: python main.py")
        print("2. Start API server: python api.py")
        print("3. Open dashboard: dashboard.html")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies first.")
        return
    
    # Create directories
    create_directories()
    
    # Ask user
    print("\n" + "="*70)
    response = input("Run quick demo? (y/n): ").strip().lower()
    
    if response == 'y':
        success = run_demo()
        if success:
            print("\n✓ System is ready to use!")
    else:
        print("\nSetup complete! Run 'python main.py' to start.")

if __name__ == "__main__":
    main()
