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
    """
    Traffic Prediction Model using Gradient Boosting.
    Drop-in replacement for the CNN-LSTM model — same API, no TensorFlow.
    """

    def __init__(self, config=None):
        self.config = config or Config()
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoders = {}
        self.model = None
        self.history = None
        self.db = TrafficDatabase()
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._train_features = None  # features used during training
        Config.create_directories()

    def prepare_data(self, df, features=None):
        """Preprocess and scale data."""
        if features is None:
            features = ['vehicle_count', 'hour', 'day_of_week',
                        'is_weekend', 'is_rush_hour']

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

        # Keep only columns that exist
        features = [f for f in features if f in df_processed.columns]
        data = df_processed[features].fillna(0).values
        scaled_data = self.scaler.fit_transform(data)
        self._train_features = features  # remember for predict_future
        return scaled_data, features

    def create_sequences(self, data, target_col=0):
        """Flatten sliding windows into feature vectors for sklearn."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            # Flatten the window into a 1-D feature vector
            X.append(data[i:i + self.sequence_length].flatten())
            y.append(data[i + self.sequence_length, target_col])
        return np.array(X), np.array(y)

    def build_model(self, input_shape=None):
        """Build a Gradient Boosting regressor."""
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        """Train the model."""
        print(f"\nTraining model...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        # Simulate history for compatibility
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
        """Make predictions. Accepts 3-D (sklearn flattens) or 2-D arrays."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X).reshape(-1, 1)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae  = mean_absolute_error(y_test, predictions)
        r2   = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100

        print(f"\nPerformance Metrics:")
        print(f"  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}  MAPE: {mape:.2f}%")

        return predictions, {'rmse': rmse, 'mae': mae, 'r2_score': r2,
                              'mape': mape, 'model_version': self.model_version}

    def predict_future(self, recent_data, hours_ahead=6):
        """Predict traffic for the next N hours."""
        df = recent_data.copy()

        # Rebuild any missing feature columns from timestamp
        if 'hour' not in df.columns and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_rush_hour'] = df['hour'].apply(
                lambda h: int((7 <= h <= 9) or (17 <= h <= 19)))

        # Use the same features the model was trained on
        features = self._train_features or ['vehicle_count', 'hour',
                                             'day_of_week', 'is_weekend', 'is_rush_hour']

        # Handle weather encoding
        if 'weather_encoded' in features:
            if 'weather_condition' in df.columns and 'weather_condition' in self.label_encoders:
                known = set(self.label_encoders['weather_condition'].classes_)
                df['weather_condition'] = df['weather_condition'].apply(
                    lambda x: x if x in known else 'Clear')
                df['weather_encoded'] = self.label_encoders['weather_condition'].transform(
                    df['weather_condition'].fillna('Clear'))
            else:
                df['weather_encoded'] = 0

        # Add any still-missing columns as zeros
        for col in features:
            if col not in df.columns:
                df[col] = 0

        data = df[features].fillna(0).values

        # Use transform (not fit_transform) to keep same scale as training
        try:
            scaled_data = self.scaler.transform(data)
        except Exception:
            # Fallback: refit if scaler shape mismatch
            scaled_data = self.scaler.fit_transform(data)

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

        # Inverse transform only the vehicle_count column (col 0)
        data_min = self.scaler.data_min_[0]
        data_max = self.scaler.data_max_[0]
        result = np.array(predictions) * (data_max - data_min) + data_min
        return result

    def save_model(self, filepath=None):
        """Save model and scaler."""
        if filepath is None:
            filepath = os.path.join(
                self.config.MODEL_DIR,
                f'traffic_model_{self.model_version}.pkl'
            )
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, filepath.replace('.pkl', '_scaler.pkl'))
        joblib.dump(self.label_encoders, filepath.replace('.pkl', '_encoders.pkl'))
        joblib.dump(self._train_features, filepath.replace('.pkl', '_features.pkl'))
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load model and scaler."""
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


    def plot_training_history(self, save=True):
        """Plot training metrics."""
        if not self.history:
            print("No training history available")
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(['Train Loss', 'Val Loss', 'Train MAE', 'Val MAE'],
               [self.history['loss'][0], self.history['val_loss'][0],
                self.history['mae'][0], self.history['val_mae'][0]],
               color=['#00ff88', '#00cc6a', '#38ef7d', '#11998e'])
        ax.set_title('Model Performance Metrics')
        ax.set_ylabel('Score')
        plt.tight_layout()
        if save:
            path = os.path.join(self.config.PLOTS_DIR, f'metrics_{self.model_version}.png')
            plt.savefig(path, dpi=150)
        plt.show()

    def plot_predictions(self, y_actual, y_pred, n_samples=200, save=True):
        """Plot actual vs predicted."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        n = min(n_samples, len(y_actual))
        axes[0].plot(y_actual[:n], label='Actual', linewidth=2)
        axes[0].plot(y_pred[:n], label='Predicted', linewidth=2, linestyle='--')
        axes[0].set_title('Actual vs Predicted Traffic')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(y_actual, y_pred, alpha=0.4, s=15)
        mn, mx = y_actual.min(), y_actual.max()
        axes[1].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
        axes[1].set_title('Scatter: Actual vs Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            path = os.path.join(self.config.PLOTS_DIR, f'predictions_{self.model_version}.png')
            plt.savefig(path, dpi=150)
        plt.show()

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
