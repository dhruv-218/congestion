# NYC Taxi Trip Congestion Prediction

This project uses machine learning to predict traffic congestion levels for NYC taxi trips and provides interactive route visualization with congestion mapping.

[View Congestion on Map](congestion_route_map.html)



## Overview

The system analyzes NYC Yellow Taxi trip data to:
- Predict congestion levels (Low, Medium, High) for new trips
- Visualize routes with color-coded congestion indicators
- Use XGBoost classifier for accurate predictions
- Provide interactive maps using Folium

## Features

- **Congestion Prediction**: ML model that predicts traffic congestion based on pickup/dropoff locations, time, and other trip features
- **Route Visualization**: Interactive maps showing predicted congestion along routes
- **Real-time Routing**: Integration with OSRM (Open Source Routing Machine) for accurate route calculation
- **Feature Engineering**: Comprehensive preprocessing including peak hour detection and categorical encoding
- **Model Persistence**: Trained models and encoders saved for reuse

## Project Structure

```
congestion/
├── full_final.ipynb          # Main Jupyter notebook with complete pipeline
├── best_xgb_model.pkl        # Trained XGBoost model
├── label_encoders.pkl        # Saved label encoders for categorical features
├── target_encoder.pkl        # Target variable encoder
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Data Requirements

The project expects the following data files (not included in repository):

1. **`yellow-tripdata-2025-01 (1).csv`** - NYC Yellow Taxi trip data
   - Required columns: `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `PULocationID`, `DOLocationID`, `VendorID`, `RatecodeID`, `store_and_fwd_flag`
   - Available from: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

2. **`location_coordinates_10000.csv`** - Location coordinates mapping
   - Required columns: `LocationID`, `Latitude`, `Longitude`
   - Maps NYC taxi zones to geographic coordinates

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd congestion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the required data files in the project directory

## Usage

### Training the Model

1. Open `full_final.ipynb` in Jupyter Notebook or VS Code
2. Run cells 1-3 to:
   - Load and preprocess the taxi trip data
   - Perform feature engineering
   - Train XGBoost model with grid search
   - Save the trained model and encoders

### Making Predictions

Run cell 4-5 to:
- Load the saved model and encoders
- Make predictions on new trip data
- Example prediction for a trip from LocationID 121 to 105

### Route Visualization

Run cells 6-7 to:
- Generate interactive route maps
- Visualize congestion levels along routes
- Save maps as HTML files

## Model Details

### Features Used
- `pickup_hour`: Hour of pickup (0-23)
- `pickup_dayofweek`: Day of week (0=Monday, 6=Sunday)
- `PULocationID`: Pickup location zone ID
- `DOLocationID`: Dropoff location zone ID
- `VendorID`: Taxi vendor identifier
- `RatecodeID`: Rate code for the trip
- `store_and_fwd_flag`: Store and forward flag
- `is_morning_peak`: Binary indicator for morning peak hours (7-10 AM on weekdays)
- `is_evening_peak`: Binary indicator for evening peak hours (4-7 PM on weekdays)

### Target Variable
- `congestion_level`: Categorical variable with 3 levels
  - 0: Low congestion
  - 1: Medium congestion  
  - 2: High congestion

### Model Performance
The XGBoost classifier uses the following optimized hyperparameters:
- Grid search performed on `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`
- Cross-validation used for model selection
- Multi-class classification with softmax probability

## Route Visualization

The system provides two types of route visualization:

1. **Basic Route Mapping**: Shows pickup/dropoff locations with route
2. **Congestion-Colored Routes**: Color-codes route segments based on estimated speed:
   - 🔴 Red: Congested (< 20 km/h)
   - 🟡 Yellow: Moderate (20-40 km/h)
   - 🟢 Green: Free flow (> 40 km/h)

## API Dependencies

- **OSRM API**: Used for route calculation
  - Endpoint: `http://router.project-osrm.org/route/v1/driving/`
  - Provides real road network routing
  - Returns GeoJSON route geometry

## Example Usage

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_xgb_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Create new trip data
new_trip = pd.DataFrame({
    'pickup_hour': [8],
    'pickup_dayofweek': [2],
    'PULocationID': ['121'],
    'DOLocationID': ['105'],
    'VendorID': ['2'],
    'RatecodeID': ['1'],
    'store_and_fwd_flag': ['N'],
    'is_morning_peak': [1],
    'is_evening_peak': [0]
})

# Encode and predict
for col in ['PULocationID', 'DOLocationID', 'VendorID', 'RatecodeID', 'store_and_fwd_flag']:
    new_trip[col] = label_encoders[col].transform(new_trip[col].astype(str))

prediction = model.predict(new_trip)
congestion_level = target_encoder.inverse_transform(prediction)[0]
print(f"Predicted congestion: {congestion_level}")
```

## Output Files

- `congestion_route_map.html`: Interactive map with route and congestion visualization
- `predicted_peakhours.csv`: Batch predictions (when processing multiple trips)

## Requirements

See `requirements.txt` for complete list of dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please ensure compliance with NYC TLC data usage terms when using taxi trip data.

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure all required CSV files are in the project directory
2. **API connectivity**: OSRM routing requires internet connection
3. **Memory issues**: Large datasets may require increased memory allocation
4. **Location ID mismatches**: Ensure location coordinates file covers all LocationIDs in trip data

### Error Handling

- The code includes error handling for missing location coordinates
- Unknown categorical values are handled by the label encoders
- OSRM API failures are caught and reported

## Future Enhancements

- Real-time traffic data integration
- Weather condition features
- Deep learning models (LSTM for temporal patterns)
- Web application interface
- Additional visualization options
- Performance optimization for large datasets
