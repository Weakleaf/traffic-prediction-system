"""
Advanced traffic data generator with realistic patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from database import TrafficDatabase
from config import Config


class TrafficDataGenerator:
    """Generate realistic traffic data for training and testing"""
    
    def __init__(self):
        self.db = TrafficDatabase()
        self.roads = Config.ROADS
    
    def generate_traffic_data(self, start_date, n_days=365, save_to_db=True, road=None):
        """
        Generate comprehensive traffic data with realistic patterns.
        If road is None, generates data for all roads.
        """
        roads_to_generate = [road] if road else self.roads
        all_data = []

        for r in roads_to_generate:
            print(f"Generating {n_days} days of traffic data for {r}...")
            data = self._generate_for_road(start_date, n_days, r, save_to_db)
            all_data.extend(data)

        df = pd.DataFrame(all_data)
        print(f"Generated {len(df)} total traffic records across {len(roads_to_generate)} road(s)")
        return df

    def _generate_for_road(self, start_date, n_days, road, save_to_db):
        """Generate hourly records for a single road."""
        n_samples = n_days * 24
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
        data = []

        # Each road has a slightly different base traffic level
        road_base = {
            'Mombasa Road': 55,
            'Thika Road': 60,
            'Kisii-Nairobi Road': 40,
            'Waiyaki Way': 50,
            'Ngong Road': 45,
            'Langata Road': 42,
            'Jogoo Road': 48,
            'Uhuru Highway': 65,
            'Outer Ring Road': 52,
            'Eastern Bypass': 38,
        }
        base_traffic = road_base.get(road, 45)

        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()
            is_weekend = day_of_week >= 5

            morning_rush = self._rush_hour_pattern(hour, 7, 9, intensity=35)
            evening_rush = self._rush_hour_pattern(hour, 17, 19, intensity=40)
            weekend_factor = 0.6 if is_weekend else 1.0
            night_reduction = 0.3 if (hour >= 23 or hour <= 5) else 1.0

            weather_conditions = ['Clear', 'Rain', 'Fog', 'Snow']
            weather_weights = [0.7, 0.2, 0.05, 0.05]
            weather = np.random.choice(weather_conditions, p=weather_weights)
            weather_impact = {'Clear': 1.0, 'Rain': 0.85, 'Fog': 0.75, 'Snow': 0.6}

            traffic = base_traffic + morning_rush + evening_rush
            traffic *= weekend_factor * night_reduction * weather_impact[weather]
            traffic += np.random.normal(0, 5)
            traffic = max(10, min(100, traffic))

            avg_speed = 60 - (traffic * 0.4) + np.random.normal(0, 3)
            avg_speed = max(10, min(80, avg_speed))
            congestion = self._determine_congestion(traffic)
            temperature = 20 + 10 * np.sin(2 * np.pi * (ts.timetuple().tm_yday / 365))
            temperature += np.random.normal(0, 3)

            record = {
                'timestamp': ts,
                'location': road,
                'road': road,
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

            if save_to_db:
                self.db.insert_traffic_record(record)

        return data
    
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
