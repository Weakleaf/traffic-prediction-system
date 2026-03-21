"""
Advanced traffic data generator with realistic patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from database import TrafficDatabase
from config import Config


class TrafficDataGenerator:
    
    def __init__(self):
        self.db = TrafficDatabase()
        self.roads = Config.ROADS
    
    def generate_traffic_data(self, start_date, n_days=365, save_to_db=True, road=None):
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
        n_samples = n_days * 24
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
        data = []
        road_base = {
            'Mombasa Road': 55, 'Thika Road': 60, 'Kisii-Nairobi Road': 40,
            'Waiyaki Way': 50, 'Ngong Road': 45, 'Langata Road': 42,
            'Jogoo Road': 48, 'Uhuru Highway': 65, 'Outer Ring Road': 52, 'Eastern Bypass': 38,
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
            weather = np.random.choice(['Clear','Rain','Fog','Snow'], p=[0.7,0.2,0.05,0.05])
            weather_impact = {'Clear': 1.0, 'Rain': 0.85, 'Fog': 0.75, 'Snow': 0.6}
            traffic = base_traffic + morning_rush + evening_rush
            traffic *= weekend_factor * night_reduction * weather_impact[weather]
            traffic += np.random.normal(0, 5)
            traffic = max(10, min(100, traffic))
            avg_speed = max(10, min(80, 60 - (traffic * 0.4) + np.random.normal(0, 3)))
            congestion = self._determine_congestion(traffic)
            temperature = round(20 + 10 * np.sin(2 * np.pi * (ts.timetuple().tm_yday / 365)) + np.random.normal(0, 3), 1)
            record = {
                'timestamp': ts, 'location': road, 'road': road,
                'vehicle_count': int(traffic), 'average_speed': round(avg_speed, 2),
                'congestion_level': congestion, 'weather_condition': weather,
                'temperature': temperature, 'hour': hour, 'day_of_week': day_of_week,
                'is_weekend': int(is_weekend), 'is_rush_hour': int(self._is_rush_hour(hour))
            }
            data.append(record)
            if save_to_db:
                self.db.insert_traffic_record(record)
        return data
    
    def _rush_hour_pattern(self, hour, peak_start, peak_end, intensity=30):
        if peak_start <= hour <= peak_end:
            peak_center = (peak_start + peak_end) / 2
            return intensity * np.exp(-abs(hour - peak_center)**2 / 2)
        return 0
    
    def _is_rush_hour(self, hour):
        return (7 <= hour <= 9) or (17 <= hour <= 19)
    
    def _determine_congestion(self, traffic):
        if traffic < 40: return 'LOW'
        elif traffic < 70: return 'MODERATE'
        elif traffic < 90: return 'HIGH'
        else: return 'SEVERE'
