"""
Test Suite for Traffic Prediction System
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

from config import Config
from model import TrafficPredictionModel
from database import TrafficDatabase
from data_generator import TrafficDataGenerator


class TestDataGenerator(unittest.TestCase):
    """Test data generation functionality"""
    
    def setUp(self):
        self.generator = TrafficDataGenerator()
    
    def test_generate_data(self):
        """Test traffic data generation"""
        df = self.generator.generate_traffic_data(
            datetime(2024, 1, 1), 
            n_days=7, 
            save_to_db=False
        )
        
        self.assertEqual(len(df), 7 * 24)  # 7 days * 24 hours
        self.assertIn('vehicle_count', df.columns)
        self.assertIn('hour', df.columns)
        self.assertIn('congestion_level', df.columns)
    
    def test_rush_hour_pattern(self):
        """Test rush hour detection"""
        self.assertTrue(self.generator._is_rush_hour(8))
        self.assertTrue(self.generator._is_rush_hour(18))
        self.assertFalse(self.generator._is_rush_hour(12))
    
    def test_congestion_levels(self):
        """Test congestion level determination"""
        self.assertEqual(self.generator._determine_congestion(30), 'LOW')
        self.assertEqual(self.generator._determine_congestion(60), 'MODERATE')
        self.assertEqual(self.generator._determine_congestion(80), 'HIGH')
        self.assertEqual(self.generator._determine_congestion(95), 'SEVERE')


class TestDatabase(unittest.TestCase):
    """Test database operations"""
    
    def setUp(self):
        self.db = TrafficDatabase(':memory:')  # Use in-memory database
    
    def test_table_creation(self):
        """Test database table creation"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('traffic_records', tables)
        self.assertIn('predictions', tables)
        self.assertIn('model_performance', tables)
        
        conn.close()
    
    def test_insert_traffic_record(self):
        """Test inserting traffic record"""
        record = {
            'timestamp': datetime.now(),
            'location': 'Test Road',
            'vehicle_count': 50,
            'average_speed': 45.5,
            'congestion_level': 'MODERATE',
            'weather_condition': 'Clear',
            'temperature': 25.0
        }
        
        self.db.insert_traffic_record(record)
        
        # Verify insertion
        df = self.db.get_traffic_data(limit=1)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['vehicle_count'], 50)


class TestModel(unittest.TestCase):
    """Test model functionality"""
    
    def setUp(self):
        self.config = Config()
        self.model = TrafficPredictionModel(self.config)
        self.generator = TrafficDataGenerator()
    
    def test_data_preparation(self):
        """Test data preprocessing"""
        df = self.generator.generate_traffic_data(
            datetime(2024, 1, 1), 
            n_days=7, 
            save_to_db=False
        )
        
        scaled_data, features = self.model.prepare_data(df)
        
        self.assertIsNotNone(scaled_data)
        self.assertTrue(len(features) > 0)
        self.assertTrue(np.all(scaled_data >= 0) and np.all(scaled_data <= 1))
    
    def test_sequence_creation(self):
        """Test time-series sequence creation"""
        df = self.generator.generate_traffic_data(
            datetime(2024, 1, 1), 
            n_days=7, 
            save_to_db=False
        )
        
        scaled_data, _ = self.model.prepare_data(df)
        X, y = self.model.create_sequences(scaled_data)
        
        self.assertEqual(X.shape[1], self.config.SEQUENCE_LENGTH)
        self.assertEqual(len(X), len(y))
        self.assertTrue(len(X) > 0)
    
    def test_model_building(self):
        """Test model architecture building"""
        input_shape = (24, 5)  # 24 time steps, 5 features
        model = self.model.build_model(input_shape)
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers) > 0, True)
    
    def test_prediction_shape(self):
        """Test prediction output shape"""
        # Create dummy data
        X_dummy = np.random.rand(10, 24, 5)
        
        # Build and compile model
        self.model.model = self.model.build_model((24, 5))
        
        # Make prediction
        predictions = self.model.predict(X_dummy)
        
        self.assertEqual(predictions.shape[0], 10)
        self.assertEqual(predictions.shape[1], 1)


class TestConfig(unittest.TestCase):
    """Test configuration"""
    
    def test_config_values(self):
        """Test configuration parameters"""
        config = Config()
        
        self.assertGreater(config.SEQUENCE_LENGTH, 0)
        self.assertGreater(config.EPOCHS, 0)
        self.assertGreater(config.BATCH_SIZE, 0)
        self.assertIsNotNone(config.FEATURES)
    
    def test_directory_creation(self):
        """Test directory creation"""
        Config.create_directories()
        
        self.assertTrue(os.path.exists(Config.DATA_DIR))
        self.assertTrue(os.path.exists(Config.MODEL_DIR))
        self.assertTrue(os.path.exists(Config.PLOTS_DIR))


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline"""
        # Generate data
        generator = TrafficDataGenerator()
        df = generator.generate_traffic_data(
            datetime(2024, 1, 1), 
            n_days=10, 
            save_to_db=False
        )
        
        # Initialize model
        model = TrafficPredictionModel()
        
        # Prepare data
        scaled_data, features = model.prepare_data(df)
        X, y = model.create_sequences(scaled_data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Build model
        model.model = model.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train (1 epoch for speed)
        model.model.fit(X_train, y_train, epochs=1, verbose=0)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Verify
        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(np.all(np.isfinite(predictions)))


def run_tests():
    """Run all tests"""
    print("="*70)
    print(" " * 20 + "RUNNING TEST SUITE")
    print("="*70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
