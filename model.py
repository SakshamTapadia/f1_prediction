import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from config import logger, FEATURE_COLS, MODEL_PARAMS
from feature_engineering import enhance_data_with_circuit_features, prepare_features_for_model
from utils import log_feature_importance, display_prediction_results

def train_comprehensive_model(race_data, quali_data, target_circuit_name=None, target_year=None):
    """Train an enhanced model with comprehensive historical data."""
    if race_data is None or quali_data is None:
        logger.error("Insufficient data to train model!")
        return None
    
    # Filter data if target circuit is specified
    if target_circuit_name:
        race_data = race_data[race_data['CircuitName'].str.contains(target_circuit_name, case=False, na=False)]
        quali_data = quali_data[quali_data['CircuitName'].str.contains(target_circuit_name, case=False, na=False)]
        logger.info(f"Filtered to {len(race_data)} race records and {len(quali_data)} qualifying records for {target_circuit_name}")
    
    # Filter out the target year for cross-validation if specified
    if target_year:
        train_race_data = race_data[race_data['Year'] != target_year]
        logger.info(f"Excluding {target_year} from training data. Training on {len(train_race_data)} race records.")
    else:
        train_race_data = race_data
    
    # Merge race and qualifying data
    combined_data = pd.merge(
        train_race_data,
        quali_data[['FullName', 'TeamName', 'Year', 'CircuitName', 'BestQualiTime', 
                   'AirTemp', 'TrackTemp', 'Humidity']],
        on=['FullName', 'TeamName', 'Year', 'CircuitName'],
        how='inner'
    )
    
    if combined_data.empty:
        logger.error("No matching data after merging race and qualifying information!")
        return None
    
    # Add circuit features
    combined_data = enhance_data_with_circuit_features(combined_data)
    
    # Handle missing values
    combined_data = combined_data.fillna({
        'AirTemp': combined_data['AirTemp'].median(),
        'TrackTemp': combined_data['TrackTemp'].median(),
        'Humidity': combined_data['Humidity'].median(),
        'BestQualiTime': combined_data['BestQualiTime'].median(),
        'LapTime (s)_mean': combined_data['LapTime (s)_mean'].median(),
        'LapTime (s)_min': combined_data['LapTime (s)_min'].median(),
        'LapTime (s)_std': combined_data['LapTime (s)_std'].median()
    })
    
    # Make sure all feature columns exist
    available_features = [col for col in FEATURE_COLS if col in combined_data.columns]
    
    # Convert boolean features to int
    for col in available_features:
        if combined_data[col].dtype == bool:
            combined_data[col] = combined_data[col].astype(int)
    
    X = combined_data[available_features]
    y = combined_data['LapTime (s)_mean']  # Using mean lap time as target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with better parameters
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Model trained with Mean Absolute Error: {mae:.2f} seconds")
    
    # Feature importance
    log_feature_importance(available_features, model.feature_importances_)
    
    return model

def predict_race_winner(model, quali_data):
    """Predict the winner based on qualifying data and additional factors."""
    if model is None or quali_data is None:
        logger.error("Cannot make prediction without model or qualifying data!")
        return None
    
    # Prepare features for prediction
    feature_cols = [col for col in FEATURE_COLS if col in quali_data.columns]
    
    # Make prediction
    quali_data['Predicted Lap Time'] = model.predict(quali_data[feature_cols])
    
    # Sort by predicted lap time (faster is better)
    sorted_predictions = quali_data.sort_values('Predicted Lap Time')
    
    # Display top 5 predicted finishers
    display_prediction_results(sorted_predictions)
    
    return sorted_predictions