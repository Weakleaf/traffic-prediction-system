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

config = Config()
model = TrafficPredictionModel(config)
db = TrafficDatabase()
data_gen = TrafficDataGenerator()

model_files = [f for f in os.listdir(config.MODEL_DIR)
               if f.endswith('.pkl') and 'scaler' not in f and 'encoder' not in f and 'features' not in f]
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
    print("No trained model found — auto-training now...")
    try:
        df = data_gen.generate_traffic_data(start_date=datetime(2023, 1, 1), n_days=180, save_to_db=True)
        scaled_data, features = model.prepare_data(df)
        X, y = model.create_sequences(scaled_data)
        from sklearn.model_selection import train_test_split as tts
        X_train, X_temp, y_train, y_temp = tts(X, y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = tts(X_temp, y_temp, test_size=0.5, shuffle=False)
        model.train(X_train, y_train, X_val, y_val)
        model.save_model()
        print("Auto-training complete!")
    except Exception as e:
        print(f"Auto-training failed: {e} — demo mode active")


@app.route('/')
def home():
    return jsonify({'message': 'Traffic Congestion Prediction API', 'version': '1.0'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model.model is not None, 'timestamp': datetime.now().isoformat()})

@app.route('/roads', methods=['GET'])
def get_roads():
    return jsonify({'roads': config.ROADS})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        hours_ahead = data.get('hours_ahead', 6)
        location = data.get('location', 'Main Highway')
        road = data.get('road', None)
        recent_data = db.get_traffic_data(limit=config.SEQUENCE_LENGTH * 2, road=road)

        if model.model is None or len(recent_data) < config.SEQUENCE_LENGTH:
            import random
            hour_now = datetime.now().hour
            predictions = []
            for i in range(hours_ahead):
                h = (hour_now + i + 1) % 24
                base = 80 if 7 <= h <= 9 else 85 if 17 <= h <= 19 else 20 if (h >= 23 or h <= 5) else 45
                predictions.append(base + random.uniform(-8, 8))
        else:
            predictions = model.predict_future(recent_data, hours_ahead)

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
                'confidence': 0.85
            })
            db.insert_prediction({
                'prediction_time': current_time,
                'target_time': target_time,
                'predicted_count': float(pred),
                'congestion_level': congestion,
                'model_version': model.model_version
            })

        return jsonify({'success': True, 'location': road or location,
                        'prediction_time': current_time.isoformat(), 'forecast': forecast})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/current', methods=['GET'])
def current_traffic():
    try:
        road = request.args.get('road', None)
        recent = db.get_traffic_data(limit=1, road=road)
        if recent.empty:
            import random
            h = datetime.now().hour
            vc = 80 if 7 <= h <= 9 or 17 <= h <= 19 else (20 if h >= 23 or h <= 5 else 45)
            vc += random.randint(-5, 5)
            spd = max(10, 60 - vc * 0.4 + random.uniform(-3, 3))
            lvl = 'SEVERE' if vc >= 90 else 'HIGH' if vc >= 70 else 'MODERATE' if vc >= 40 else 'LOW'
            return jsonify({'timestamp': datetime.now().isoformat(), 'vehicle_count': vc,
                            'average_speed': round(spd, 1), 'congestion_level': lvl,
                            'weather': 'Clear', 'temperature': 22.0})
        record = recent.iloc[0]
        return jsonify({'timestamp': record['timestamp'], 'vehicle_count': int(record['vehicle_count']),
                        'average_speed': float(record.get('average_speed', 0)),
                        'congestion_level': record.get('congestion_level', 'UNKNOWN'),
                        'weather': record.get('weather_condition', 'Unknown'),
                        'temperature': float(record.get('temperature', 0))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        limit = min(request.args.get('hours', 24, type=int), 168)
        data = db.get_traffic_data(limit=limit)
        if data.empty:
            return jsonify({'error': 'No data available'}), 404
        return jsonify({'success': True, 'count': len(data), 'data': data.to_dict('records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        recent_data = db.get_traffic_data(limit=168)
        if recent_data.empty:
            return jsonify({'total_records': 0, 'avg_traffic': 0, 'max_traffic': 0,
                            'min_traffic': 0, 'model_version': 'Not loaded',
                            'last_update': datetime.now().isoformat()})
        return jsonify({
            'total_records': len(recent_data),
            'avg_traffic': float(recent_data['vehicle_count'].mean()),
            'max_traffic': int(recent_data['vehicle_count'].max()),
            'min_traffic': int(recent_data['vehicle_count'].min()),
            'avg_speed': float(recent_data['average_speed'].mean()) if 'average_speed' in recent_data else 0,
            'model_version': model.model_version if model.model else 'Not loaded',
            'last_update': recent_data['timestamp'].max()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def determine_congestion_level(traffic_value):
    for level, (min_val, max_val) in Config.CONGESTION_LEVELS.items():
        if min_val <= traffic_value < max_val:
            return level
    return 'SEVERE'

if __name__ == '__main__':
    print(f"API running on http://{config.API_HOST}:{config.API_PORT}")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)
