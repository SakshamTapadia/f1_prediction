import warnings
import pandas as pd
from datetime import datetime
from config import logger
import os
import pandas as pd
import fastf1
from datetime import datetime
import numpy as np

from config import logger, FIRST_F1_YEAR, CURRENT_YEAR, ALL_RACE_DATA_PATH, ALL_QUALI_DATA_PATH, GRAND_PRIX_NAMES
from data_processor import preprocess_race_data, preprocess_quali_data
from feature_engineering import enhance_data_with_circuit_features

def load_or_build_comprehensive_data():
    """Load comprehensive dataset from disk, or build it if not available."""
    # Try to load from disk
    if os.path.exists(ALL_RACE_DATA_PATH) and os.path.exists(ALL_QUALI_DATA_PATH):
        logger.info("Loading data from cached files...")
        race_data = pd.read_csv(ALL_RACE_DATA_PATH)
        quali_data = pd.read_csv(ALL_QUALI_DATA_PATH)
        
        if not race_data.empty and not quali_data.empty:
            logger.info(f"Loaded {len(race_data)} race records and {len(quali_data)} qualifying records from cache")
            return race_data, quali_data
    
    # Build from scratch
    logger.info("Building comprehensive F1 dataset...")
    race_data_list = []
    quali_data_list = []
    
    # Get all seasons
    first_year = FIRST_F1_YEAR
    current_year = CURRENT_YEAR
    seasons = range(first_year, current_year + 1)
    
    total_combinations = len(seasons) * len(GRAND_PRIX_NAMES)
    success_count = 0
    combination_count = 0
    
    # Iterate through seasons and events
    for year in seasons:
        for gp_name in GRAND_PRIX_NAMES:
            combination_count += 1
            display_progress(combination_count, total_combinations, success_count, interval=5)
            
            try:
                # Try to load race session
                race_session = get_race_data(year, gp_name, "R")
                quali_session = get_race_data(year, gp_name, "Q")
                
                # Process race data
                if race_session is not None:
                    race_data = preprocess_race_data(race_session)
                    if race_data is not None and not race_data.empty:
                        # Add circuit features
                        race_data = enhance_data_with_circuit_features(race_data)
                        race_data_list.append(race_data)
                
                # Process qualifying data
                if quali_session is not None:
                    quali_data = preprocess_quali_data(quali_session)
                    if quali_data is not None and not quali_data.empty:
                        # Add circuit features
                        quali_data = enhance_data_with_circuit_features(quali_data)
                        quali_data_list.append(quali_data)
                
                if race_session is not None or quali_session is not None:
                    success_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {year} {gp_name}: {e}")
    
    # Combine all data
    if race_data_list:
        all_race_data = pd.concat(race_data_list, ignore_index=True)
        all_race_data.to_csv(ALL_RACE_DATA_PATH, index=False)
        logger.info(f"Saved {len(all_race_data)} race records to {ALL_RACE_DATA_PATH}")
    else:
        all_race_data = None
        logger.warning("No race data collected!")
    
    if quali_data_list:
        all_quali_data = pd.concat(quali_data_list, ignore_index=True)
        all_quali_data.to_csv(ALL_QUALI_DATA_PATH, index=False)
        logger.info(f"Saved {len(all_quali_data)} qualifying records to {ALL_QUALI_DATA_PATH}")
    else:
        all_quali_data = None
        logger.warning("No qualifying data collected!")
    
    return all_race_data, all_quali_data

def get_race_data(year, grand_prix, session_type="R"):
    """Load race or qualifying session data safely."""
    try:
        # Try to load the session
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()
        return session
    except Exception as e:
        logger.debug(f"Could not load {session_type} session for {year} {grand_prix}: {e}")
        return None

def get_current_quali_data(year, grand_prix):
    """Get qualifying data for prediction."""
    try:
        # Load qualifying session
        quali_session = get_race_data(year, grand_prix, "Q")
        if quali_session is None:
            logger.error(f"No qualifying data found for {year} {grand_prix}")
            return None
        
        # Process qualifying data
        quali_data = preprocess_quali_data(quali_session)
        if quali_data is None or quali_data.empty:
            logger.error("Failed to process qualifying data")
            return None
        
        # Add circuit features
        enhanced_quali = enhance_data_with_circuit_features(quali_data)
        
        return enhanced_quali
        
    except Exception as e:
        logger.error(f"Error getting qualifying data: {e}")
        return None

def get_circuit_specific_data(all_race_data, all_quali_data, circuit_identifier):
    """Filter data specific to a circuit."""
    if all_race_data is None or all_quali_data is None:
        return None, None
    
    # Get circuit-specific data
    circuit_race_data = all_race_data[all_race_data['CircuitName'].str.contains(circuit_identifier, case=False, na=False)]
    circuit_quali_data = all_quali_data[all_quali_data['CircuitName'].str.contains(circuit_identifier, case=False, na=False)]
    
    logger.info(f"Found {len(circuit_race_data)} race and {len(circuit_quali_data)} qualifying records for {circuit_identifier}")
    
    return circuit_race_data, circuit_quali_data

def suppress_warnings():
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

def get_all_f1_seasons(first_year, current_year=None):
    """Get a list of all F1 seasons from the start of available data to current year."""
    if current_year is None:
        current_year = datetime.now().year
    return range(first_year, current_year + 1)

def display_progress(current, total, success_count, interval=10):
    """Display progress information at specified intervals."""
    if current % interval == 0:
        logger.info(f"Progress: {current}/{total} combinations checked, {success_count} successful")

def display_prediction_results(sorted_predictions):
    """Display top 5 predicted finishers."""
    logger.info("\n🏁 Predicted Race Order:")
    for i, (_, driver) in enumerate(sorted_predictions[['FullName', 'TeamName', 'Predicted Lap Time']].iterrows(), 1):
        logger.info(f"{i}. {driver['FullName']} ({driver['TeamName']}) - {driver['Predicted Lap Time']:.3f}s")
    
    winner = sorted_predictions.iloc[0]
    logger.info(f"\n🏆 Predicted Winner: {winner['FullName']} ({winner['TeamName']})")

def display_comparison_results(prediction, actual_race_data):
    """Display comparison between predicted and actual results."""
    if prediction is None or actual_race_data is None:
        logger.error("Cannot compare with actual results - missing data!")
        return None
    
    # Get actual finishing order
    actual_order = actual_race_data.sort_values('Position')
    
    # Get predicted order
    predicted_order = prediction.sort_values('Predicted Lap Time')
    
    # Compare top positions
    logger.info("\n📊 Prediction Accuracy:")
    logger.info("Pos | Predicted         | Actual")
    logger.info("----|------------------|------------------")
    
    accuracy_count = 0
    for pos in range(min(10, len(actual_order), len(predicted_order))):
        predicted_driver = predicted_order.iloc[pos]['FullName']
        
        if pos < len(actual_order):
            actual_driver = actual_order.iloc[pos]['FullName']
            match = predicted_driver == actual_driver
            if match:
                accuracy_count += 1
            logger.info(f"{pos+1:3d} | {predicted_driver:18s} | {actual_driver:18s} {'✓' if match else '✗'}")
    
    top10_accuracy = accuracy_count / min(10, len(actual_order)) * 100
    logger.info(f"\nTop 10 Accuracy: {top10_accuracy:.1f}%")
    
    return top10_accuracy

def log_feature_importance(feature_cols, importance_values):
    """Log feature importance in a readable format."""
    feature_importance = sorted(zip(feature_cols, importance_values), 
                               key=lambda x: x[1], reverse=True)
    logger.info("Feature importance:")
    for feature, importance in feature_importance:
        logger.info(f"  - {feature}: {importance:.4f}")