"""
Advanced traffic data generator with realistic patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from database import TrafficDatabase


class TrafficDataGenerator:
    """Generate realistic traffic data for training and testing"""
    
    def __init__(self):
        self.db = TrafficDatabase()
    
    def generate_traffic_data(self, start_date, n_days=365, save_to_db=True):
        """
        Generate comprehensive traffic data with realistic patterns
        
        Features:
        - Rush hour patterns (morning 7-9am, evening 5-7pm)
        - Weekend reduction
        - Weather impact
        - Special events
        - Seasonal variations
        """
        print(f"Generating {n_days} days of traffic data...")
        
        n_samples = n_days * 24  # Hourly data
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
        
        data = []
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day_of_week = ts.weekday()
            is_weekend = day_of_week >= 5
            
            # Base traffic
            base_traffic = 45
            
            # Rush hour patterns
            morning_rush = self._rush_hour_pattern(hour, 7, 9, intensity=35)
            evening_rush = self._rush_hour_pattern(hour, 17, 19, intensity=40)
            
            # Weekend reduction
            weekend_factor = 0.6 if is_weekend else 1.0
            
            # Time of day pattern
            night_reduction = 0.3 if (hour >= 23 or hour <= 5) else 1.0
            
            # Weather simulation (random)
            weather_conditions = ['Clear', 'Rain', 'Fog', 'Snow']
            weather_weights = [0.7, 0.2, 0.05, 0.05]
            weather = np.random.choice(weather_conditions, p=weather_weights)
            
            weather_impact = {
                'Clear': 1.0,
                'Rain': 0.85,
                'Fog': 0.75,
                'Snow': 0.6
            }
            
            # Calculate vehicle count
            traffic = base_traffic + morning_rush + evening_rush
            traffic *= weekend_factor * night_reduction * weather_impact[weather]
            
            # Add noise
            traffic += np.random.normal(0, 5)
            traffic = max(10, min(100, traffic))  # Clamp between 10-100
            
            # Average speed (inversely related to traffic)
            avg_speed = 60 - (traffic * 0.4) + np.random.normal(0, 3)
            avg_speed = max(10, min(80, avg_speed))
            
            # Congestion level
            congestion = self._determine_congestion(traffic)
            
            # Temperature
            temperature = 20 + 10 * np.sin(2 * np.pi * (ts.timetuple().tm_yday / 365))
            temperature += np.random.normal(0, 3)
            
            record = {
                'timestamp': ts,
                'location': 'Main Highway',
                'vehicle_count': int(traffic),
                'average_speed': round(avg_speed, 2),
                'congestion_level': congestion,
                'weather_condition': weather,
                'temperature': round(temperature, 1),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': int(is_weekend),
                'is_rush_hour': int(self._is_rush_hour(hour))
            }
            
            data.append(record)
            
            # Save to database
            if save_to_db and i % 100 == 0:
                self.db.insert_traffic_record(record)
        
        df = pd.DataFrame(data)
        
        print(f"Generated {len(df)} traffic records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def _rush_hour_pattern(self, hour, peak_start, peak_end, intensity=30):
        """Calculate rush hour traffic increase"""
        if peak_start <= hour <= peak_end:
            # Gaussian-like peak
            peak_center = (peak_start + peak_end) / 2
            distance = abs(hour - peak_center)
            return intensity * np.exp(-distance**2 / 2)
        return 0
    
    def _is_rush_hour(self, hour):
        """Check if hour is rush hour"""
        return (7 <= hour <= 9) or (17 <= hour <= 19)
    
    def _determine_congestion(self, traffic):
        """Determine congestion level based on traffic volume"""
        if traffic < 40:
            return 'LOW'
        elif traffic < 70:
            return 'MODERATE'
        elif traffic < 90:
            return 'HIGH'
        else:
            return 'SEVERE'
    
    def load_or_generate_data(self, filepath='data/traffic_data.csv', 
                             n_days=365, force_generate=False):
        """Load existing data or generate new"""
        import os
        
        if os.path.exists(filepath) and not force_generate:
            print(f"Loading existing data from {filepath}")
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
        else:
            print("Generating new traffic data...")
            start_date = datetime(2023, 1, 1)
            df = self.generate_traffic_data(start_date, n_days, save_to_db=False)
            
            # Save to CSV
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
        
        return df
