import fastf1
from datetime import datetime
import traceback

from config import logger, CACHE_DIR
from utils import suppress_warnings, display_comparison_results
from data_loader import (
    load_or_build_comprehensive_data, 
    get_circuit_specific_data, 
    get_race_data, 
    get_current_quali_data
)
from model import train_comprehensive_model, predict_race_winner

def main():
    """Main function to run the comprehensive F1 prediction model."""
    # Suppress warnings for cleaner output
    suppress_warnings()
    
    # Enable fastF1 cache
    fastf1.Cache.enable_cache(CACHE_DIR)
    
    # Get user input
    try:
        year = int(input("Enter race year to predict: "))
        grand_prix = input("Enter Grand Prix name (e.g., 'Australian Grand Prix'): ")
    except ValueError:
        logger.error("Invalid year input. Please enter a valid integer year.")
        return
    
    # For validation, check if race has already happened
    current_date = datetime.now()
    race_already_happened = (year < current_date.year) or \
                           (year == current_date.year and input("Has this race already occurred? (y/n): ").lower() == 'y')
    
    if race_already_happened:
        actual_winner = input("Enter the actual winner (for validation): ")
        logger.info(f"Note: The actual winner of {year} {grand_prix} was {actual_winner}")
    
    # Load or build comprehensive dataset
    logger.info("\n📊 Loading historical F1 data...")
    all_race_data, all_quali_data = load_or_build_comprehensive_data()
    
    if all_race_data is None or all_quali_data is None:
        logger.error("Failed to get historical data. Exiting.")
        return
    
    # Get circuit-specific data (optional)
    circuit_identifier = grand_prix.split(' ')[0]  # Use first word of GP name as circuit identifier
    circuit_race_data, circuit_quali_data = get_circuit_specific_data(all_race_data, all_quali_data, circuit_identifier)
    
    if circuit_race_data is not None and not circuit_race_data.empty:
        logger.info(f"Using circuit-specific model for {circuit_identifier}")
        training_race_data, training_quali_data = circuit_race_data, circuit_quali_data
    else:
        logger.info("Using general model with all historical data")
        training_race_data, training_quali_data = all_race_data, all_quali_data
    
    # Train model excluding target year (for validation)
    logger.info("\n🔧 Training prediction model...")
    model = train_comprehensive_model(
        training_race_data, 
        training_quali_data, 
        target_circuit_name=circuit_identifier,
        target_year=year if race_already_happened else None
    )
    
    if model is None:
        logger.error("Failed to train model. Exiting.")
        return
    
    # Get current qualifying data
    logger.info(f"\n🏎️ Getting qualifying data for {grand_prix} {year}...")
    current_quali = get_current_quali_data(year, grand_prix)
    
    if current_quali is None:
        logger.error(f"No qualifying data available for {grand_prix} {year}!")
        return
    
    # Make prediction
    logger.info("\n🔮 Making race prediction...")
    prediction = predict_race_winner(model, current_quali)
    
    if prediction is None:
        logger.error("Failed to make prediction. Exiting.")
        return
    
    # If race has happened, compare with actual results
    if race_already_happened:
        try:
            actual_race_session = get_race_data(year, grand_prix, "R")
            if actual_race_session is not None:
                display_comparison_results(prediction, actual_race_session.results)
            else:
                logger.warning("Could not load actual race results for comparison")
        except Exception as e:
            logger.error(f"Error comparing with actual results: {e}")
    
    # Save prediction to file
    prediction_file = f"prediction_{year}_{grand_prix.replace(' ', '_')}.csv"
    prediction.to_csv(prediction_file, index=False)
    logger.info(f"Prediction saved to {prediction_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in the main program: {e}")
        logger.error(traceback.format_exc())