# Getting Started Guide

## Traffic Congestion Prediction System
**CYBER TITANS - Kisii University 2026**

---

## Quick Start (5 Minutes)

### Option 1: Automated Setup

```bash
# Run the quick start script
python quick_start.py
```

This will:
- Check all dependencies
- Create necessary directories
- Run a quick demo
- Train a small model
- Verify everything works

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create directories
mkdir data models logs plots

# 3. Run the main system
python main.py

# 4. Start the API (in another terminal)
python api.py

# 5. Open dashboard.html in your browser
```

---

## Step-by-Step Tutorial

### Step 1: Installation

Make sure you have Python 3.8+ installed:

```bash
python --version
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Step 2: Understanding the System

The system has 4 main components:

1. **Data Layer** (`data_generator.py`, `database.py`)
   - Generates realistic traffic data
   - Stores data in SQLite database

2. **Model Layer** (`model.py`)
   - CNN-LSTM hybrid architecture
   - Training and prediction logic

3. **API Layer** (`api.py`)
   - REST API for predictions
   - Endpoints for data access

4. **Application Layer** (`main.py`, `dashboard.html`)
   - Main training script
   - Web dashboard for visualization

### Step 3: Generate Data

The system can generate synthetic traffic data:

```python
from data_generator import TrafficDataGenerator
from datetime import datetime

generator = TrafficDataGenerator()
df = generator.generate_traffic_data(
    start_date=datetime(2023, 1, 1),
    n_days=365,
    save_to_db=True
)
```

### Step 4: Train the Model

Run the complete training pipeline:

```bash
python main.py
```

This will:
1. Generate/load traffic data
2. Preprocess and normalize
3. Create time-series sequences
4. Train CNN-LSTM model
5. Evaluate performance
6. Save model and visualizations

Expected output:
```
Training samples:   5,000 (70.0%)
Validation samples: 1,000 (15.0%)
Test samples:       1,000 (15.0%)

Epoch 1/30
157/157 [==============================] - 15s 95ms/step
...
Final RMSE: 0.0523
Final MAE:  0.0412
```

### Step 5: Start the API

In a new terminal:

```bash
python api.py
```

The API will start on `http://localhost:5000`

### Step 6: Test the API

**Using cURL:**

```bash
# Health check
curl http://localhost:5000/health

# Get current traffic
curl http://localhost:5000/current

# Get prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"hours_ahead": 6}'
```

**Using Python:**

```python
import requests

# Get 6-hour forecast
response = requests.post('http://localhost:5000/predict', 
                        json={'hours_ahead': 6})
data = response.json()

for hour in data['forecast']:
    print(f"Hour {hour['hour']}: {hour['predicted_traffic']:.1f} - {hour['congestion_level']}")
```

### Step 7: Use the Dashboard

1. Open `dashboard.html` in your web browser
2. View current traffic status
3. Get predictions for future hours
4. Monitor statistics

---

## Common Tasks

### Task 1: Retrain the Model

```bash
python main.py
```

### Task 2: Use a Trained Model

```python
from model import TrafficPredictionModel

# Load existing model
model = TrafficPredictionModel()
model.load_model('models/traffic_model_v20260309_120000.h5')

# Make prediction
predictions = model.predict(X_test)
```

### Task 3: Add Real Data

```python
from database import TrafficDatabase
from datetime import datetime

db = TrafficDatabase()

# Insert real traffic record
record = {
    'timestamp': datetime.now(),
    'location': 'Highway A1',
    'vehicle_count': 75,
    'average_speed': 35.5,
    'congestion_level': 'HIGH',
    'weather_condition': 'Rain',
    'temperature': 18.5
}

db.insert_traffic_record(record)
```

### Task 4: Customize Configuration

Edit `config.py`:

```python
class Config:
    SEQUENCE_LENGTH = 48  # Use 48 hours of history
    EPOCHS = 100          # Train for more epochs
    BATCH_SIZE = 64       # Larger batch size
```

### Task 5: Run Tests

```bash
python test_system.py
```

---

## Troubleshooting

### Problem: Import errors

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Problem: Model not loading in API

**Solution:**
```bash
# Train a new model first
python main.py

# Then start API
python api.py
```

### Problem: Database errors

**Solution:**
```python
# Reset database
import os
os.remove('data/traffic_data.db')

# Recreate
from database import TrafficDatabase
db = TrafficDatabase()
```

### Problem: Low prediction accuracy

**Solutions:**
1. Generate more training data (increase n_days)
2. Increase training epochs
3. Adjust model architecture in `model.py`
4. Add more features to the dataset

### Problem: API connection refused

**Solution:**
```bash
# Check if API is running
curl http://localhost:5000/health

# If not, start it
python api.py
```

---

## Performance Optimization

### For Better Accuracy:

1. **More Data**: Generate 2-3 years of data
   ```python
   df = generator.generate_traffic_data(start_date, n_days=1095)
   ```

2. **Longer Training**: Increase epochs
   ```python
   model.train(X_train, y_train, X_val, y_val, epochs=100)
   ```

3. **Feature Engineering**: Add more features
   - Day of month
   - Month of year
   - Holiday indicators
   - Special events

### For Faster Training:

1. **Reduce Data**: Use fewer days
2. **Smaller Batch Size**: Reduce memory usage
3. **Fewer Epochs**: Quick testing
4. **GPU Acceleration**: Use TensorFlow with GPU

---

## Next Steps

### Beginner:
1. ✅ Run quick_start.py
2. ✅ Explore the dashboard
3. ✅ Test API endpoints
4. ✅ Read the code comments

### Intermediate:
1. ✅ Modify model architecture
2. ✅ Add new features
3. ✅ Integrate real data sources
4. ✅ Customize visualizations

### Advanced:
1. ✅ Implement Graph Neural Networks
2. ✅ Add Transformer models
3. ✅ Deploy to cloud (AWS, Azure, GCP)
4. ✅ Build mobile app
5. ✅ Real-time sensor integration

---

## Resources

### Documentation:
- `README.md` - Complete system overview
- `config.py` - Configuration options
- Code comments - Inline documentation

### Learning:
- TensorFlow: https://www.tensorflow.org/tutorials
- LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- CNN: https://cs231n.github.io/

### Support:
- Check `test_system.py` for examples
- Review error messages carefully
- Contact CYBER TITANS team

---

## Project Structure Reference

```
traffic-prediction-system/
│
├── main.py                 # Main training script
├── api.py                  # REST API server
├── model.py                # CNN-LSTM model
├── config.py               # Configuration
├── database.py             # Database operations
├── data_generator.py       # Data generation
├── quick_start.py          # Quick setup script
├── test_system.py          # Test suite
├── dashboard.html          # Web dashboard
├── requirements.txt        # Dependencies
├── README.md              # Full documentation
└── GETTING_STARTED.md     # This file
```

---

## Success Checklist

- [ ] Dependencies installed
- [ ] Directories created
- [ ] Data generated
- [ ] Model trained
- [ ] API running
- [ ] Dashboard accessible
- [ ] Tests passing
- [ ] Predictions working

---

**Congratulations! You're ready to predict traffic congestion! 🚦**

For questions or issues, refer to README.md or contact the CYBER TITANS team.

© 2026 CYBER TITANS - Kisii University
