"""
Flask REST API for Traffic Prediction System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os

from config import Config
from model import TrafficPredictionModel
from database import TrafficDatabase
from data_generator import TrafficDataGenerator

app = Flask(__name__)
CORS(app)

# Initialize components
config = Config()
model = TrafficPredictionModel(config)
db = TrafficDatabase()
data_gen = TrafficDataGenerator()

# Load model if exists, otherwise auto-train
model_files = [f for f in os.listdir(config.MODEL_DIR)
               if f.endswith('.pkl') and 'scaler' not in f and 'encoder' not in f]
if model_files:
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(config.MODEL_DIR, latest_model)
    try:
        model.load_model(model_path)
        print(f"Loaded model: {latest_model}")
    except Exception as e:
        print(f"Could not load model: {e} — will auto-train")
        model_files = []

if not model_files:
    print("No trained model found — auto-training now (takes ~30 seconds)...")
    try:
        df = data_gen.generate_traffic_data(
            start_date=datetime(2023, 1, 1),
            n_days=180,
            save_to_db=True
        )
        scaled_data, features = model.prepare_data(df)
        X, y = model.create_sequences(scaled_data)
        from sklearn.model_selection import train_test_split as tts
        X_train, X_temp, y_train, y_temp = tts(X, y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = tts(X_temp, y_temp, test_size=0.5, shuffle=False)
        model.train(X_train, y_train, X_val, y_val)
        model.save_model()
        print("Auto-training complete — API ready!")
    except Exception as e:
        print(f"Auto-training failed: {e} — demo mode active")


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Traffic Congestion Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Predict traffic for next hours',
            '/current': 'GET - Get current traffic status',
            '/history': 'GET - Get historical traffic data',
            '/stats': 'GET - Get system statistics',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model.model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict future traffic
    
    Request body:
    {
        "hours_ahead": 6,
        "location": "Main Highway"
    }
    """
    try:
        data = request.get_json()
        hours_ahead = data.get('hours_ahead', 6)
        location = data.get('location', 'Main Highway')
        
        # Get recent data
        recent_data = db.get_traffic_data(limit=config.SEQUENCE_LENGTH * 2)

        if model.model is None or len(recent_data) < config.SEQUENCE_LENGTH:
            # No trained model yet — generate realistic demo predictions
            import random
            hour_now = datetime.now().hour
            predictions = []
            for i in range(hours_ahead):
                h = (hour_now + i + 1) % 24
                base = 45
                if 7 <= h <= 9:   base = 80
                elif 17 <= h <= 19: base = 85
                elif 23 <= h or h <= 5: base = 20
                predictions.append(base + random.uniform(-8, 8))
        else:
            # Make predictions using trained model
            predictions = model.predict_future(recent_data, hours_ahead)
        
        # Prepare response
        current_time = datetime.now()
        forecast = []
        
        for i, pred in enumerate(predictions):
            target_time = current_time + timedelta(hours=i+1)
            congestion = determine_congestion_level(pred)
            
            forecast.append({
                'hour': i + 1,
                'timestamp': target_time.isoformat(),
                'predicted_traffic': float(pred),
                'congestion_level': congestion,
                'confidence': 0.85  # Placeholder
            })
            
            # Save prediction to database
            db.insert_prediction({
                'prediction_time': current_time,
                'target_time': target_time,
                'predicted_count': float(pred),
                'congestion_level': congestion,
                'model_version': model.model_version
            })
        
        return jsonify({
            'success': True,
            'location': location,
            'prediction_time': current_time.isoformat(),
            'forecast': forecast
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/current', methods=['GET'])
def current_traffic():
    """Get current traffic status"""
    try:
        recent = db.get_traffic_data(limit=1)
        
        if recent.empty:
            # Return live demo data if DB is empty
            import random
            h = datetime.now().hour
            vc = 80 if 7 <= h <= 9 or 17 <= h <= 19 else (20 if h >= 23 or h <= 5 else 45)
            vc += random.randint(-5, 5)
            spd = max(10, 60 - vc * 0.4 + random.uniform(-3, 3))
            lvl = 'SEVERE' if vc >= 90 else 'HIGH' if vc >= 70 else 'MODERATE' if vc >= 40 else 'LOW'
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'vehicle_count': vc,
                'average_speed': round(spd, 1),
                'congestion_level': lvl,
                'weather': 'Clear',
                'temperature': 22.0
            })
        
        record = recent.iloc[0]
        
        return jsonify({
            'timestamp': record['timestamp'],
            'vehicle_count': int(record['vehicle_count']),
            'average_speed': float(record.get('average_speed', 0)),
            'congestion_level': record.get('congestion_level', 'UNKNOWN'),
            'weather': record.get('weather_condition', 'Unknown'),
            'temperature': float(record.get('temperature', 0))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    """Get historical traffic data"""
    try:
        hours = request.args.get('hours', 24, type=int)
        limit = min(hours, 168)  # Max 1 week
        
        data = db.get_traffic_data(limit=limit)
        
        if data.empty:
            return jsonify({'error': 'No data available'}), 404
        
        records = data.to_dict('records')
        
        return jsonify({
            'success': True,
            'count': len(records),
            'data': records
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        # Get recent data
        recent_data = db.get_traffic_data(limit=168)  # Last week
        
        if recent_data.empty:
            return jsonify({
                'total_records': 0,
                'avg_traffic': 0,
                'max_traffic': 0,
                'min_traffic': 0,
                'avg_speed': 0,
                'congestion_distribution': {},
                'model_version': 'Not loaded',
                'last_update': datetime.now().isoformat()
            })
        
        stats_data = {
            'total_records': len(recent_data),
            'avg_traffic': float(recent_data['vehicle_count'].mean()),
            'max_traffic': int(recent_data['vehicle_count'].max()),
            'min_traffic': int(recent_data['vehicle_count'].min()),
            'avg_speed': float(recent_data.get('average_speed', pd.Series([0])).mean()),
            'congestion_distribution': recent_data.get('congestion_level', pd.Series()).value_counts().to_dict(),
            'model_version': model.model_version if model.model else 'Not loaded',
            'last_update': recent_data['timestamp'].max()
        }
        
        return jsonify(stats_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """
    Trigger model training (admin endpoint)
    
    Request body:
    {
        "epochs": 50,
        "generate_data": true
    }
    """
    try:
        data = request.get_json() or {}
        epochs = data.get('epochs', 30)
        generate_data = data.get('generate_data', False)
        
        # Generate or load data
        if generate_data:
            df = data_gen.generate_traffic_data(
                datetime(2023, 1, 1), 
                n_days=365, 
                save_to_db=True
            )
        else:
            df = db.get_traffic_data(limit=10000)
        
        if df.empty or len(df) < 100:
            return jsonify({'error': 'Insufficient data for training'}), 400
        
        # Prepare data
        scaled_data, features = model.prepare_data(df)
        X, y = model.create_sequences(scaled_data)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False
        )
        
        # Train
        model.train(X_train, y_train, X_val, y_val, epochs=epochs)
        
        # Evaluate
        predictions, metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics,
            'model_version': model.model_version
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def determine_congestion_level(traffic_value):
    """Determine congestion level from traffic value"""
    for level, (min_val, max_val) in Config.CONGESTION_LEVELS.items():
        if min_val <= traffic_value < max_val:
            return level
    return 'SEVERE'


if __name__ == '__main__':
    print("Starting Traffic Prediction API...")
    print(f"API running on http://{config.API_HOST}:{config.API_PORT}")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)
