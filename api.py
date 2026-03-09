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

# Load model if exists
model_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('.h5')]
if model_files:
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(config.MODEL_DIR, latest_model)
    try:
        model.load_model(model_path)
        print(f"Loaded model: {latest_model}")
    except Exception as e:
        print(f"Could not load model: {e}")


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
        
        if model.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get recent data
        recent_data = db.get_traffic_data(limit=config.SEQUENCE_LENGTH * 2)
        
        if len(recent_data) < config.SEQUENCE_LENGTH:
            return jsonify({
                'error': f'Insufficient data. Need at least {config.SEQUENCE_LENGTH} records'
            }), 400
        
        # Make predictions
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
            return jsonify({'error': 'No traffic data available'}), 404
        
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
            return jsonify({'error': 'No data available'}), 404
        
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
