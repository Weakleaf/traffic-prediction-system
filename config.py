"""
Configuration settings for Traffic Prediction System
"""

import os

class Config:
    # Model parameters
    SEQUENCE_LENGTH = 24  # Hours of historical data to use
    PREDICTION_HORIZON = 6  # Hours ahead to predict
    
    # CNN-LSTM Architecture
    CNN_FILTERS = [64, 32]
    CNN_KERNEL_SIZE = 3
    LSTM_UNITS = [50, 50]
    DENSE_UNITS = [25]
    DROPOUT_RATE = 0.2
    
    # Training parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    LEARNING_RATE = 0.001
    
    # Data parameters
    FEATURES = ['vehicle_count', 'hour', 'day_of_week', 'is_weekend', 
                'is_rush_hour', 'weather_condition']
    TARGET = 'vehicle_count'
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
    
    # Database
    DATABASE_PATH = os.path.join(DATA_DIR, 'traffic_data.db')
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    DEBUG = True
    
    # Congestion thresholds
    CONGESTION_LEVELS = {
        'LOW': (0, 40),
        'MODERATE': (40, 70),
        'HIGH': (70, 90),
        'SEVERE': (90, 100)
    }
    
    @staticmethod
    def create_directories():
        """Create necessary directories"""
        for directory in [Config.DATA_DIR, Config.MODEL_DIR, 
                         Config.LOGS_DIR, Config.PLOTS_DIR]:
            os.makedirs(directory, exist_ok=True)
