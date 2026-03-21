"""
Traffic Prediction Model — scikit-learn based (no TensorFlow required)
Uses a RandomForest + feature engineering pipeline to simulate CNN-LSTM predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime

from config import Config
from database import TrafficDatabase


class TrafficPredictionModel:

    def __init__(self, config=None):
        self.config = config or Config()
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoders = {}
        self.model = None
        self.history = None
        self.db = TrafficDatabase()
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Config.create_directories()

    def prepare_data(self, df, features=None):
        if features is None:
            features = ['vehicle_count', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
        df_processed = df.copy()
        if 'weather_condition' in df.columns:
            if 'weather_condition' not in self.label_encoders:
                self.label_encoders['weather_condition'] = LabelEncoder()
                df_processed['weather_encoded'] = self.label_encoders['weather_condition'].fit_transform(
                    df['weather_condition'].fillna('Clear'))
            else:
                df_processed['weather_encoded'] = self.label_encoders['weather_condition'].transform(
                    df['weather_condition'].fillna('Clear'))
            if 'weather_encoded' not in features:
                features = list(features) + ['weather_encoded']
        features = [f for f in features if f in df_processed.columns]
        data = df_processed[features].fillna(0).values
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data, features

    def create_sequences(self, data, target_col=0):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length].flatten())
            y.append(data[i + self.sequence_length, target_col])
        return np.array(X), np.array(y)

    def build_model(self, input_shape=None):
        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42, verbose=0)

    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        print(f"\nTraining model...")
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        self.history = {
            'loss': [mean_squared_error(y_train, train_pred)],
            'val_loss': [mean_squared_error(y_val, val_pred)],
            'mae': [mean_absolute_error(y_train, train_pred)],
            'val_mae': [mean_absolute_error(y_val, val_pred)],
        }
        print("Training completed!")
        self._save_training_metrics(len(X_train), 1)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X).reshape(-1, 1)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae  = mean_absolute_error(y_test, predictions)
        r2   = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100
        return predictions, {'rmse': rmse, 'mae': mae, 'r2_score': r2,
                              'mape': mape, 'model_version': self.model_version}

    def predict_future(self, recent_data, hours_ahead=6):
        scaled_data, _ = self.prepare_data(recent_data)
        if len(scaled_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} records")
        predictions = []
        current_seq = scaled_data[-self.sequence_length:].copy()
        for _ in range(hours_ahead):
            X = current_seq.flatten().reshape(1, -1)
            pred = self.model.predict(X)[0]
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1, 0] = pred
        dummy = np.zeros((len(predictions), scaled_data.shape[1]))
        dummy[:, 0] = predictions
        return self.scaler.inverse_transform(dummy)[:, 0]

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, f'traffic_model_{self.model_version}.pkl')
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, filepath.replace('.pkl', '_scaler.pkl'))
        joblib.dump(self.label_encoders, filepath.replace('.pkl', '_encoders.pkl'))
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        scaler_path   = filepath.replace('.pkl', '_scaler.pkl')
        enc_path      = filepath.replace('.pkl', '_encoders.pkl')
        features_path = filepath.replace('.pkl', '_features.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(enc_path):
            self.label_encoders = joblib.load(enc_path)
        if os.path.exists(features_path):
            self._train_features = joblib.load(features_path)
        print(f"Model loaded from: {filepath}")

    def _save_training_metrics(self, samples_trained, epochs):
        if not self.history:
            return
        self.db.save_model_performance({
            'model_version': self.model_version,
            'rmse': np.sqrt(self.history['val_loss'][0]),
            'mae': self.history['val_mae'][0],
            'r2_score': 0.0,
            'samples_trained': samples_trained,
            'epochs': epochs,
            'notes': f'GradientBoosting model — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        })
