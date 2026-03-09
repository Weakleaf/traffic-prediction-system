"""
Database management for Traffic Prediction System
"""

import sqlite3
import pandas as pd
from datetime import datetime
from config import Config


class TrafficDatabase:
    """Manage traffic data storage and retrieval"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.create_tables()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def create_tables(self):
        """Create database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Traffic records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                location TEXT,
                vehicle_count INTEGER,
                average_speed REAL,
                congestion_level TEXT,
                weather_condition TEXT,
                temperature REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time DATETIME NOT NULL,
                target_time DATETIME NOT NULL,
                predicted_count REAL,
                actual_count REAL,
                congestion_level TEXT,
                confidence REAL,
                model_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                rmse REAL,
                mae REAL,
                r2_score REAL,
                training_date DATETIME,
                samples_trained INTEGER,
                epochs INTEGER,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        print("Database tables created successfully")
    
    def insert_traffic_record(self, record):
        """Insert traffic record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO traffic_records 
            (timestamp, location, vehicle_count, average_speed, 
             congestion_level, weather_condition, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (record['timestamp'], record.get('location', 'Main Road'),
              record['vehicle_count'], record.get('average_speed'),
              record.get('congestion_level'), record.get('weather_condition'),
              record.get('temperature')))
        
        conn.commit()
        conn.close()
    
    def insert_prediction(self, prediction):
        """Insert prediction record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (prediction_time, target_time, predicted_count, actual_count,
             congestion_level, confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (prediction['prediction_time'], prediction['target_time'],
              prediction['predicted_count'], prediction.get('actual_count'),
              prediction['congestion_level'], prediction.get('confidence', 0.95),
              prediction.get('model_version', 'v1.0')))
        
        conn.commit()
        conn.close()
    
    def get_traffic_data(self, start_date=None, end_date=None, limit=None):
        """Retrieve traffic data"""
        conn = self.get_connection()
        
        query = "SELECT * FROM traffic_records"
        params = []
        
        if start_date:
            query += " WHERE timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            if start_date:
                query += " AND timestamp <= ?"
            else:
                query += " WHERE timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_predictions(self, limit=100):
        """Retrieve predictions"""
        conn = self.get_connection()
        df = pd.read_sql_query(
            f"SELECT * FROM predictions ORDER BY prediction_time DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return df
    
    def save_model_performance(self, metrics):
        """Save model performance metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_version, rmse, mae, r2_score, training_date, 
             samples_trained, epochs, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (metrics['model_version'], metrics['rmse'], metrics['mae'],
              metrics.get('r2_score'), datetime.now(),
              metrics.get('samples_trained'), metrics.get('epochs'),
              metrics.get('notes')))
        
        conn.commit()
        conn.close()
