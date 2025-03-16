import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("F1Predictor")

# Cache directory setup
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Define the first year of modern F1 data availability in FastF1
FIRST_F1_YEAR = 2018  # FastF1 has reliable data from around 2018 onwards

# Current year
CURRENT_YEAR = datetime.now().year

# File paths
ALL_RACE_DATA_PATH = "all_race_data.csv"
ALL_QUALI_DATA_PATH = "all_quali_data.csv"

# Default values for missing data
DEFAULT_VALUES = {
    'AirTemp': 25.0,
    'TrackTemp': 30.0,
    'Humidity': 50.0,
    'BestQualiTime': 90.0
}

# List of all Grand Prix names to search historical data
GRAND_PRIX_NAMES = [
    "Australian Grand Prix", "Bahrain Grand Prix", "Vietnamese Grand Prix",
    "Chinese Grand Prix", "Dutch Grand Prix", "Spanish Grand Prix",
    "Monaco Grand Prix", "Azerbaijan Grand Prix", "Canadian Grand Prix",
    "French Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Italian Grand Prix", 
    "Singapore Grand Prix", "Russian Grand Prix", "Japanese Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "Brazilian Grand Prix",
    "Abu Dhabi Grand Prix", "Miami Grand Prix", "Emilia Romagna Grand Prix",
    "Portuguese Grand Prix", "Styrian Grand Prix", "70th Anniversary Grand Prix",
    "Tuscan Grand Prix", "Sakhir Grand Prix", "Saudi Arabian Grand Prix",
    "Qatar Grand Prix", "Las Vegas Grand Prix", "São Paulo Grand Prix"
]

# Circuit characteristics dictionaries
STREET_CIRCUITS = ['monaco', 'singapore', 'baku', 'jeddah', 'las vegas']
HIGH_SPEED_CIRCUITS = ['monza', 'spa', 'silverstone', 'saudi', 'baku']
HIGH_DOWNFORCE_CIRCUITS = ['hungary', 'monaco', 'singapore']
HIGH_ALTITUDE_CIRCUITS = ['mexico', 'brazil', 'austria']
HIGH_TEMP_CIRCUITS = ['bahrain', 'singapore', 'abu dhabi', 'saudi']
WET_PRONE_CIRCUITS = ['spa', 'brazil', 'japan']

# Model feature columns
FEATURE_COLS = [
    'BestQualiTime', 'AirTemp', 'TrackTemp', 'Humidity', 'Year',
    'is_street_circuit', 'is_high_speed', 'is_high_downforce',
    'is_high_altitude', 'is_high_temp', 'is_wet_prone'
]

# XGBoost model parameters
MODEL_PARAMS = {
    'n_estimators': 150,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}