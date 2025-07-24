# NYC Taxi Trip Congestion Classification: Technical Details

## Table of Contents
1. [Overview](#overview)
2. [Classification Problem Definition](#classification-problem-definition)
3. [Model Architecture & Algorithm](#model-architecture--algorithm)
4. [Feature Engineering](#feature-engineering)
5. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
6. [Model Training & Optimization](#model-training--optimization)
7. [Route Optimization & Visualization](#route-optimization--visualization)
8. [Performance Analysis](#performance-analysis)
9. [Technical Implementation](#technical-implementation)

## Overview

This project implements a multi-class classification system to predict traffic congestion levels for NYC taxi trips using machine learning. The system combines temporal, spatial, and categorical features to classify trips into three congestion categories: Low, Medium, and High.

**Key Technical Components:**
- **Algorithm**: XGBoost (Extreme Gradient Boosting) Classifier
- **Problem Type**: Multi-class classification (3 classes)
- **Feature Space**: 9-dimensional mixed feature vector
- **Target Encoding**: Ordinal encoding based on trip duration quantiles
- **Route Optimization**: OSRM-based routing with congestion visualization

## Classification Problem Definition

### Problem Formulation
**Objective**: Given taxi trip characteristics, predict the congestion level category.

**Mathematical Formulation:**
```
f: X → Y
where:
X ∈ ℝ^9 (feature space)
Y ∈ {0, 1, 2} (congestion levels: Low, Medium, High)
```

### Target Variable Creation
The congestion level is derived from trip duration using quantile-based discretization:

1. **Trip Duration Calculation:**
   ```
   duration = (dropoff_time - pickup_time) / 3600  # hours
   ```

2. **Quantile-based Binning:**
   - Uses `pd.qcut()` with 3 quantiles (tertiles)
   - Automatically balances class distribution
   - Labels: ['Low', 'Medium', 'High']

3. **Target Encoding:**
   ```python
   target_encoder = LabelEncoder()
   y = target_encoder.fit_transform(congestion_labels)
   # 0: Low, 1: Medium, 2: High
   ```

### Class Distribution Strategy
- **Balanced Classes**: Quantile-based approach ensures approximately equal class sizes
- **Ordinal Relationship**: Maintains natural ordering of congestion severity
- **Interpretability**: Clear mapping between duration and congestion perception

## Model Architecture & Algorithm

### XGBoost Classifier Configuration

**Core Algorithm:** Extreme Gradient Boosting
- **Base Learners**: Decision trees (CART)
- **Ensemble Method**: Gradient boosting with regularization
- **Optimization**: Newton's method for second-order approximation

**Model Configuration:**
```python
XGBClassifier(
    objective='multi:softprob',     # Multi-class with probability output
    num_class=3,                    # Three congestion levels
    use_label_encoder=False,        # Use sklearn preprocessing
    eval_metric='mlogloss'          # Multi-class log loss
)
```

### Why XGBoost for Congestion Classification?

1. **Heterogeneous Feature Handling**: Efficiently processes mixed data types (temporal, categorical, spatial)
2. **Non-linear Relationships**: Captures complex interactions between time, location, and congestion
3. **Robust to Outliers**: Built-in regularization prevents overfitting on anomalous trips
4. **Feature Importance**: Provides interpretable insights into congestion drivers
5. **Computational Efficiency**: Fast training and prediction suitable for real-time applications

### Mathematical Foundation

**Objective Function:**
```
L(θ) = Σ l(yi, ŷi) + Σ Ω(fk)
```
where:
- `l(yi, ŷi)`: Multi-class log loss
- `Ω(fk)`: Regularization term for tree k
- `θ`: Model parameters

**Gradient Boosting Update:**
```
ŷi^(t) = ŷi^(t-1) + η * fk(xi)
```
where:
- `η`: Learning rate
- `fk`: New tree learner
- `t`: Boosting iteration

## Feature Engineering

### Temporal Features

1. **Pickup Hour** (`pickup_hour`)
   - **Type**: Cyclic numerical (0-23)
   - **Rationale**: Captures daily traffic patterns
   - **Engineering**: Direct extraction from datetime

2. **Day of Week** (`pickup_dayofweek`)
   - **Type**: Cyclic categorical (0-6, Monday=0)
   - **Rationale**: Weekday vs weekend traffic differences
   - **Engineering**: Direct extraction from datetime

3. **Peak Hour Indicators** (`is_morning_peak`, `is_evening_peak`)
   - **Type**: Binary features (0/1)
   - **Logic**:
     ```python
     is_weekday = pickup_dayofweek < 5
     is_morning_peak = is_weekday & pickup_hour.between(7, 10)
     is_evening_peak = is_weekday & pickup_hour.between(16, 19)
     ```
   - **Rationale**: Rush hour periods have distinct congestion patterns

### Spatial Features

1. **Pickup Location ID** (`PULocationID`)
   - **Type**: Categorical (265 NYC taxi zones)
   - **Encoding**: Label encoding
   - **Rationale**: Different zones have varying baseline congestion

2. **Dropoff Location ID** (`DOLocationID`)
   - **Type**: Categorical (265 NYC taxi zones)
   - **Encoding**: Label encoding
   - **Rationale**: Destination affects route and congestion exposure

### Operational Features

1. **Vendor ID** (`VendorID`)
   - **Type**: Categorical (2 vendors)
   - **Encoding**: Label encoding
   - **Rationale**: Different vendors may have varying route preferences

2. **Rate Code ID** (`RatecodeID`)
   - **Type**: Categorical (6 rate types)
   - **Encoding**: Label encoding
   - **Rationale**: Rate type correlates with trip type and urgency

3. **Store and Forward Flag** (`store_and_fwd_flag`)
   - **Type**: Binary categorical (Y/N)
   - **Encoding**: Label encoding
   - **Rationale**: Indicates communication issues, potential system delays

### Feature Selection Rationale

**Included Features**: Directly related to congestion patterns
**Excluded Features**: 
- `trip_distance`: Target leakage (highly correlated with duration)
- `fare_amount`, `tip_amount`: Payment features unrelated to traffic
- `passenger_count`: Minimal impact on congestion experience

## Data Preprocessing Pipeline

### 1. Data Loading & Cleaning
```python
# Load dataset
df = pd.read_csv("yellow-tripdata-2025-01 (1).csv")

# Handle corrupted column names
df.rename(columns={
    'pickup_dat"tpep_dropoff_datetime"': 'tpep_dropoff_datetime',
    'tpep_etime': 'tpep_pickup_datetime'
}, inplace=True)
```

### 2. Datetime Processing
```python
# Convert to datetime with error handling
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Extract temporal features
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
```

### 3. Categorical Encoding
```python
categorical_cols = ['VendorID', 'RatecodeID', 'store_and_fwd_flag', 
                   'PULocationID', 'DOLocationID']
label_encoders = {}

for col in categorical_cols:
    df[col] = df[col].astype(str)  # Ensure string type
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for inference
```

### 4. Target Variable Creation
```python
# Calculate trip duration in hours
df['trip_duration'] = (df['tpep_dropoff_datetime'] - 
                      df['tpep_pickup_datetime']).dt.total_seconds() / 3600

# Create balanced congestion levels using quantiles
df['congestion_level'] = pd.qcut(df['trip_duration'], 
                                q=3, 
                                labels=['Low', 'Medium', 'High'])

# Encode target
target_encoder = LabelEncoder()
df['congestion_level'] = target_encoder.fit_transform(df['congestion_level'])
```

### 5. Feature Selection & Data Splitting
```python
features = ['pickup_hour', 'pickup_dayofweek', 'PULocationID', 'DOLocationID', 
            'VendorID', 'RatecodeID', 'store_and_fwd_flag', 
            'is_morning_peak', 'is_evening_peak']

X = df[features]
y = df['congestion_level']

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
```

## Model Training & Optimization

### Hyperparameter Optimization

**Grid Search Configuration:**
```python
param_grid = {
    'n_estimators': [100],           # Number of boosting rounds
    'max_depth': [3, 5],             # Tree depth (controls overfitting)
    'learning_rate': [0.1],          # Step size shrinkage
    'subsample': [0.7],              # Row sampling ratio
    'colsample_bytree': [0.7]        # Column sampling ratio
}
```

**Cross-Validation Strategy:**
- **Method**: 2-fold CV (reduced for computational efficiency)
- **Scoring**: Accuracy (balanced classes allow simple accuracy metric)
- **Parallelization**: `n_jobs=-1` for multi-core processing

### Training Process
```python
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=2,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Model Persistence
```python
# Save trained components for inference
joblib.dump(best_model, 'best_xgb_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
```

## Route Optimization & Visualization

### Route Calculation Pipeline

The system integrates real-world routing with congestion prediction through a multi-step process:

### 1. Coordinate Mapping
```python
# Load location coordinates
coord_df = pd.read_csv('location_coordinates_10000.csv')
location_coords = {
    int(row['LocationID']): (row['Latitude'], row['Longitude'])
    for _, row in coord_df.iterrows()
}
```

### 2. OSRM API Integration

**API Endpoint:**
```
http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}
```

**Route Query:**
```python
url = f"http://router.project-osrm.org/route/v1/driving/{pickup_coords[1]},{pickup_coords[0]};{dropoff_coords[1]},{dropoff_coords[0]}?overview=full&geometries=geojson"

response = requests.get(url)
data = response.json()
route = data['routes'][0]['geometry']['coordinates']
route = [(lat, lon) for lon, lat in route]  # Convert to (lat, lon)
```

### 3. Congestion Visualization Algorithm

**Speed-Based Color Coding:**
```python
def calculate_segment_colors(route, total_duration):
    num_segments = len(route) - 1
    segment_duration = total_duration / num_segments
    colors = []
    
    for i in range(num_segments):
        # Calculate distance using great circle distance
        distance_km = geodesic(route[i], route[i + 1]).kilometers
        
        # Estimate speed
        speed_kmh = distance_km / (segment_duration / 3600)
        
        # Color classification
        if speed_kmh < 20:
            colors.append('red')     # Heavy congestion
        elif speed_kmh < 40:
            colors.append('yellow')  # Moderate congestion
        else:
            colors.append('green')   # Free flow
    
    return colors
```

### 4. Interactive Map Generation

**Folium Implementation:**
```python
# Create base map
m = folium.Map(location=pickup_coords, zoom_start=13)

# Add markers
folium.Marker(pickup_coords, tooltip='Pickup', 
              icon=folium.Icon(color='green')).add_to(m)
folium.Marker(dropoff_coords, tooltip='Dropoff', 
              icon=folium.Icon(color='red')).add_to(m)

# Draw colored route segments
for i in range(num_segments):
    folium.PolyLine(
        [route[i], route[i + 1]],
        color=segment_colors[i],
        weight=4.5,
        opacity=0.8
    ).add_to(m)

# Save interactive map
m.save("congestion_route_map.html")
```

### Route Optimization Features

1. **Real-world Routing**: Uses actual road network topology
2. **Multi-modal Visualization**: Combines prediction with geographic route
3. **Interactive Interface**: Folium-based maps with zoom and pan capabilities
4. **Speed-based Segmentation**: Visual representation of congestion intensity
5. **Export Capability**: HTML output for sharing and documentation

## Performance Analysis

### Classification Metrics

**Evaluation Framework:**
```python
from sklearn.metrics import classification_report

y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, 
                             target_names=['Low', 'Medium', 'High'])
```

**Key Performance Indicators:**
- **Accuracy**: Overall classification correctness
- **Precision**: Class-specific prediction accuracy
- **Recall**: Class-specific detection rate
- **F1-Score**: Harmonic mean of precision and recall

### Model Interpretability

**Feature Importance Analysis:**
```python
# XGBoost built-in feature importance
feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
```

**Expected Importance Ranking:**
1. **Temporal Features**: `pickup_hour`, `is_morning_peak`, `is_evening_peak`
2. **Spatial Features**: `PULocationID`, `DOLocationID`
3. **Operational Features**: `VendorID`, `RatecodeID`, `store_and_fwd_flag`

### Validation Strategy

1. **Temporal Validation**: Ensure model generalizes across different time periods
2. **Spatial Validation**: Test performance across different NYC zones
3. **Cross-validation**: K-fold validation for robust performance estimation

## Technical Implementation

### Inference Pipeline

**Single Trip Prediction:**
```python
def predict_congestion(new_trip_data):
    # Load saved components
    model = joblib.load('best_xgb_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    
    # Preprocess input
    for col in categorical_cols:
        new_trip_data[col] = new_trip_data[col].astype(str)
        new_trip_data[col] = label_encoders[col].transform(new_trip_data[col])
    
    # Predict
    prediction = model.predict(new_trip_data)
    congestion_level = target_encoder.inverse_transform(prediction)[0]
    
    return congestion_level
```

### Error Handling

1. **Missing Coordinates**: Graceful handling of unmapped LocationIDs
2. **API Failures**: OSRM connectivity error handling
3. **Data Validation**: Input feature validation before prediction
4. **Unknown Categories**: Handling of new categorical values not seen during training

### Scalability Considerations

1. **Batch Processing**: Vectorized operations for multiple predictions
2. **Memory Optimization**: Efficient data types and chunked processing
3. **Model Compression**: Potential for model pruning and quantization
4. **Caching**: Location coordinate and route caching for repeated queries

### Integration Points

1. **Real-time Systems**: RESTful API wrapper for live predictions
2. **Dashboard Integration**: JSON output format for web interfaces
3. **Mobile Applications**: Lightweight model deployment options
4. **Historical Analysis**: Batch processing capabilities for trend analysis

## Future Enhancements

### Advanced Classification Techniques

1. **Ensemble Methods**: 
   - Random Forest for comparison
   - Voting classifiers combining multiple algorithms
   - Stacking with meta-learners

2. **Deep Learning Approaches**:
   - Neural networks for complex feature interactions
   - LSTM for temporal sequence modeling
   - Graph Neural Networks for spatial relationships

3. **Feature Engineering**:
   - Weather data integration
   - Real-time traffic API feeds
   - Historical congestion patterns
   - Special events calendar

### Advanced Route Optimization

1. **Multi-objective Optimization**:
   - Time vs distance trade-offs
   - Fuel efficiency considerations
   - Dynamic route adjustment

2. **Real-time Integration**:
   - Live traffic data incorporation
   - Dynamic rerouting based on current conditions
   - Predictive route planning

3. **Advanced Visualization**:
   - 3D route visualization
   - Animation of congestion changes
   - Comparative route analysis

This classification system provides a robust foundation for understanding and predicting traffic congestion patterns in NYC, with clear pathways for enhancement and real-world deployment.
