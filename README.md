# Traffic Congestion Prediction System

**Using CNN-LSTM Machine Learning Architecture**

## Authors: CYBER TITANS
- JOB NYOLEI - IN17/00083/23
- JUSTUS YONGO - IN17/00106/23
- WYCLIFFE NZILU - IN17/00014/23
- SAMMY OWAKA - IN17/00050/21
- SOLOMON KAMAU - IN17/00036/23
- BENSON THANG'WA - IN17/00009/23

**Institution:** Kisii University  
**Year:** 2026

---

## Overview

This system predicts traffic congestion using a hybrid CNN-LSTM deep learning architecture. It combines:
- **CNN layers** for spatial feature extraction
- **LSTM layers** for temporal pattern learning
- **Real-time prediction API**
- **Comprehensive visualization**
- **Database integration**

## Features

✅ Advanced CNN-LSTM hybrid architecture  
✅ Real-time traffic prediction (1-12 hours ahead)  
✅ RESTful API for integration  
✅ Comprehensive data visualization  
✅ SQLite database for data persistence  
✅ Model versioning and performance tracking  
✅ Realistic traffic data generation  
✅ Multiple evaluation metrics (RMSE, MAE, R², MAPE)

## System Architecture

```
┌─────────────────┐
│  Data Layer     │
│  - Generator    │
│  - Database     │
└────────┬────────┘
         │
┌────────▼────────┐
│  Model Layer    │
│  - CNN-LSTM     │
│  - Training     │
│  - Prediction   │
└────────┬────────┘
         │
┌────────▼────────┐
│  API Layer      │
│  - Flask REST   │
│  - Endpoints    │
└────────┬────────┘
         │
┌────────▼────────┐
│  Client Layer   │
│  - Dashboard    │
│  - Visualization│
└─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone or download the project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories**
```bash
mkdir data models logs plots
```

## Usage

### 1. Train the Model

Run the main training script:

```bash
python main.py
```

This will:
- Generate synthetic traffic data (or load existing)
- Preprocess and normalize data
- Train the CNN-LSTM model
- Evaluate performance
- Save the trained model
- Generate visualizations

### 2. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:5000`

### 3. API Endpoints

#### Get Prediction
```bash
POST /predict
Content-Type: application/json

{
  "hours_ahead": 6,
  "location": "Main Highway"
}
```

#### Get Current Traffic
```bash
GET /current
```

#### Get Historical Data
```bash
GET /history?hours=24
```

#### Get Statistics
```bash
GET /stats
```

#### Health Check
```bash
GET /health
```

### 4. Example API Usage

**Python:**
```python
import requests

# Get prediction
response = requests.post('http://localhost:5000/predict', 
                        json={'hours_ahead': 6})
forecast = response.json()

print(f"Next 6 hours forecast:")
for hour in forecast['forecast']:
    print(f"Hour {hour['hour']}: {hour['predicted_traffic']:.1f} vehicles - {hour['congestion_level']}")
```

**cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"hours_ahead": 6}'
```

## Project Structure

```
traffic-prediction-system/
│
├── main.py                 # Main training script
├── api.py                  # Flask REST API
├── model.py                # CNN-LSTM model implementation
├── config.py               # Configuration settings
├── database.py             # Database management
├── data_generator.py       # Traffic data generator
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── data/                  # Data storage
│   ├── traffic_data.csv
│   └── traffic_data.db
│
├── models/                # Saved models
│   ├── traffic_model_*.h5
│   ├── *_scaler.pkl
│   └── *_encoders.pkl
│
├── plots/                 # Visualizations
│   ├── training_history_*.png
│   ├── predictions_*.png
│   └── traffic_analysis.png
│
└── logs/                  # System logs
```

## Model Architecture

### CNN-LSTM Hybrid Network

```
Input (24 hours × features)
    ↓
Conv1D (64 filters, kernel=3) + ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.2)
    ↓
Conv1D (32 filters, kernel=3) + ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.2)
    ↓
LSTM (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM (50 units)
    ↓
Dropout (0.2)
    ↓
Dense (25 units) + ReLU
    ↓
Dropout (0.2)
    ↓
Dense (1 unit) - Output
```

## Performance Metrics

The system evaluates model performance using:

- **RMSE** (Root Mean Square Error): Measures prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Model fit quality
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## Configuration

Edit `config.py` to customize:

```python
# Model parameters
SEQUENCE_LENGTH = 24        # Hours of history to use
PREDICTION_HORIZON = 6      # Hours ahead to predict
EPOCHS = 50                 # Training epochs
BATCH_SIZE = 32            # Batch size

# Congestion thresholds
CONGESTION_LEVELS = {
    'LOW': (0, 40),
    'MODERATE': (40, 70),
    'HIGH': (70, 90),
    'SEVERE': (90, 100)
}
```

## Data Features

The system uses the following features:

- **vehicle_count**: Number of vehicles detected
- **hour**: Hour of day (0-23)
- **day_of_week**: Day of week (0-6)
- **is_weekend**: Weekend indicator
- **is_rush_hour**: Rush hour indicator
- **weather_condition**: Weather (Clear, Rain, Fog, Snow)
- **temperature**: Temperature in Celsius
- **average_speed**: Average vehicle speed

## Database Schema

### traffic_records
- timestamp, location, vehicle_count, average_speed
- congestion_level, weather_condition, temperature

### predictions
- prediction_time, target_time, predicted_count
- actual_count, congestion_level, confidence, model_version

### model_performance
- model_version, rmse, mae, r2_score
- training_date, samples_trained, epochs

## Troubleshooting

### Model not loading
```bash
# Retrain the model
python main.py
```

### API connection errors
```bash
# Check if API is running
curl http://localhost:5000/health
```

### Insufficient data error
```bash
# Generate more data
python -c "from data_generator import TrafficDataGenerator; TrafficDataGenerator().generate_traffic_data(datetime(2023,1,1), 365, True)"
```

## Future Enhancements

- [ ] Real-time data integration from traffic sensors
- [ ] Multi-location support
- [ ] Weather API integration
- [ ] Mobile application
- [ ] Advanced visualization dashboard
- [ ] Graph Neural Networks for road network modeling
- [ ] Transformer-based models

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning. Nature, 521(7553), 436-444.

## License

This project is developed for academic purposes at Kisii University.

## Contact

For questions or support, contact the CYBER TITANS team at Kisii University.

---

**© 2026 CYBER TITANS - Kisii University**
