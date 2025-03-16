import warnings
import pandas as pd
from datetime import datetime
from config import logger

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