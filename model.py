"""
Advanced CNN-LSTM Model for Traffic Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import os
from datetime import datetime

from config import Config
from database import TrafficDatabase


class TrafficPredictionModel:
    """
    Advanced CNN-LSTM Traffic Prediction Model
    
    Architecture:
    1. CNN layers: Extract spatial features from traffic sequences
    2. LSTM layers: Learn temporal dependencies
    3. Dense layers: Final prediction
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
        
        Config.create_directories()
    
    def prepare_data(self, df, features=None):
        """
        Prepare and preprocess data for training
        
        Steps:
        1. Feature selection
        2. Encoding categorical variables
        3. Normalization
        4. Sequence creation
        """
        print("Preparing data...")
        
        if features is None:
            features = ['vehicle_count', 'hour', 'day_of_week', 
                       'is_weekend', 'is_rush_hour']
        
        # Handle categorical features
        df_processed = df.copy()
        
        # Encode weather if present
        if 'weather_condition' in df.columns:
            if 'weather_condition' not in self.label_encoders:
                self.label_encoders['weather_condition'] = LabelEncoder()
                df_processed['weather_encoded'] = self.label_encoders['weather_condition'].fit_transform(
                    df['weather_condition']
                )
            else:
                df_processed['weather_encoded'] = self.label_encoders['weather_condition'].transform(
                    df['weather_condition']
                )
            features.append('weather_encoded')
        
        # Select features
        data = df_processed[features].values
        
        # Normalize
        scaled_data = self.scaler.fit_transform(data)
        
        print(f"Data shape after preprocessing: {scaled_data.shape}")
        return scaled_data, features
    
    def create_sequences(self, data, target_col=0):
        """
        Create time-series sequences for training
        
        Args:
            data: Normalized data array
            target_col: Column index to predict (default: 0 for vehicle_count)
        
        Returns:
            X: Input sequences (samples, sequence_length, features)
            y: Target values (samples,)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_col])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
        return X, y
    
    def build_model(self, input_shape):
        """
        Build CNN-LSTM hybrid architecture
        
        Architecture:
        - Conv1D layers: Extract local patterns
        - MaxPooling: Reduce dimensionality
        - LSTM layers: Capture temporal dependencies
        - Dense layers: Final prediction
        """
        print("Building CNN-LSTM model...")
        
        model = Sequential([
            # First CNN block
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=input_shape, padding='same', name='conv1'),
            MaxPooling1D(pool_size=2, name='pool1'),
            Dropout(self.config.DROPOUT_RATE, name='dropout1'),
            
            # Second CNN block
            Conv1D(filters=32, kernel_size=3, activation='relu', 
                   padding='same', name='conv2'),
            MaxPooling1D(pool_size=2, name='pool2'),
            Dropout(self.config.DROPOUT_RATE, name='dropout2'),
            
            # First LSTM layer
            LSTM(50, return_sequences=True, name='lstm1'),
            Dropout(self.config.DROPOUT_RATE, name='dropout3'),
            
            # Second LSTM layer
            LSTM(50, return_sequences=False, name='lstm2'),
            Dropout(self.config.DROPOUT_RATE, name='dropout4'),
            
            # Dense layers
            Dense(25, activation='relu', name='dense1'),
            Dropout(self.config.DROPOUT_RATE, name='dropout5'),
            Dense(1, name='output')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        """
        Train the CNN-LSTM model with callbacks
        """
        if epochs is None:
            epochs = self.config.EPOCHS
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        print(f"\nTraining model for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model if not exists
        if self.model is None:
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.config.MODEL_DIR, f'best_model_{self.model_version}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        # Save training metrics
        self._save_training_metrics(len(X_train), epochs)
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Metrics:
        - RMSE: Root Mean Square Error
        - MAE: Mean Absolute Error
        - R²: Coefficient of determination
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        predictions = self.predict(X_test)
        predictions = predictions.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100
        
        print(f"\nPerformance Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print("="*60)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'model_version': self.model_version
        }
        
        return predictions, metrics
    
    def predict_future(self, recent_data, hours_ahead=6):
        """
        Predict traffic for future hours
        
        Args:
            recent_data: Recent traffic data (DataFrame)
            hours_ahead: Number of hours to predict
        
        Returns:
            predictions: Array of predicted values
        """
        # Prepare recent data
        scaled_data, _ = self.prepare_data(recent_data)
        
        # Get last sequence
        if len(scaled_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} hours of data")
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:].copy()
        
        for _ in range(hours_ahead):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Predict next value
            pred = self.predict(X)[0, 0]
            predictions.append(pred)
            
            # Update sequence (shift and add prediction)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # Update vehicle_count
        
        # Inverse transform predictions
        dummy = np.zeros((len(predictions), scaled_data.shape[1]))
        dummy[:, 0] = predictions
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predictions_original
    
    def save_model(self, filepath=None):
        """Save model and preprocessing objects"""
        if filepath is None:
            filepath = os.path.join(
                self.config.MODEL_DIR, 
                f'traffic_model_{self.model_version}.h5'
            )
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save scaler and encoders
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        encoders_path = filepath.replace('.h5', '_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)
        
        print(f"\nModel saved to: {filepath}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Encoders saved to: {encoders_path}")
    
    def load_model(self, filepath):
        """Load model and preprocessing objects"""
        self.model = load_model(filepath)
        
        # Load scaler and encoders
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        encoders_path = filepath.replace('.h5', '_encoders.pkl')
        self.label_encoders = joblib.load(encoders_path)
        
        print(f"Model loaded from: {filepath}")
    
    def plot_training_history(self, save=True):
        """Visualize training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], linewidth=2, color='green')
            axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary text
        final_loss = self.history.history['val_loss'][-1]
        final_mae = self.history.history['val_mae'][-1]
        axes[1, 1].text(0.1, 0.5, 
                       f"Training Summary\n\n"
                       f"Final Validation Loss: {final_loss:.4f}\n"
                       f"Final Validation MAE: {final_mae:.4f}\n"
                       f"Total Epochs: {len(self.history.history['loss'])}\n"
                       f"Model Version: {self.model_version}",
                       fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.config.PLOTS_DIR, 
                                   f'training_history_{self.model_version}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training history saved to: {filepath}")
        
        plt.show()
    
    def plot_predictions(self, y_actual, y_pred, n_samples=200, save=True):
        """Visualize predictions vs actual values"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series comparison
        indices = range(min(n_samples, len(y_actual)))
        axes[0].plot(indices, y_actual[:n_samples], label='Actual Traffic', 
                    linewidth=2, alpha=0.8)
        axes[0].plot(indices, y_pred[:n_samples], label='Predicted Traffic', 
                    linewidth=2, linestyle='--', alpha=0.8)
        axes[0].set_title('Traffic Prediction: Actual vs Predicted', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Steps (Hours)')
        axes[0].set_ylabel('Normalized Traffic Volume')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_actual, y_pred, alpha=0.5, s=20)
        axes[1].plot([y_actual.min(), y_actual.max()], 
                    [y_actual.min(), y_actual.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_title('Prediction Accuracy Scatter Plot', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.config.PLOTS_DIR, 
                                   f'predictions_{self.model_version}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to: {filepath}")
        
        plt.show()
    
    def _save_training_metrics(self, samples_trained, epochs):
        """Save training metrics to database"""
        if self.history is None:
            return
        
        metrics = {
            'model_version': self.model_version,
            'rmse': np.sqrt(self.history.history['val_loss'][-1]),
            'mae': self.history.history['val_mae'][-1],
            'r2_score': 0.0,  # Will be updated after evaluation
            'samples_trained': samples_trained,
            'epochs': epochs,
            'notes': f'CNN-LSTM model trained on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        }
        
        self.db.save_model_performance(metrics)
