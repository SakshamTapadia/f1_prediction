from config import (
    STREET_CIRCUITS, HIGH_SPEED_CIRCUITS, HIGH_DOWNFORCE_CIRCUITS,
    HIGH_ALTITUDE_CIRCUITS, HIGH_TEMP_CIRCUITS, WET_PRONE_CIRCUITS
)

def extract_circuit_features(circuit_name):
    """Extract features specific to a circuit based on its name."""
    circuit_name_lower = circuit_name.lower()
    
    features = {
        'is_street_circuit': any(track in circuit_name_lower for track in STREET_CIRCUITS),
        'is_high_speed': any(track in circuit_name_lower for track in HIGH_SPEED_CIRCUITS),
        'is_high_downforce': any(track in circuit_name_lower for track in HIGH_DOWNFORCE_CIRCUITS),
        'is_high_altitude': any(track in circuit_name_lower for track in HIGH_ALTITUDE_CIRCUITS),
        'is_high_temp': any(track in circuit_name_lower for track in HIGH_TEMP_CIRCUITS),
        'is_wet_prone': any(track in circuit_name_lower for track in WET_PRONE_CIRCUITS)
    }
    return features

def enhance_data_with_circuit_features(data):
    """Add circuit-specific features to the dataset."""
    if data is None:
        return None
    
    # Extract features for each circuit
    for feature_name in [
        'is_street_circuit', 'is_high_speed', 'is_high_downforce',
        'is_high_altitude', 'is_high_temp', 'is_wet_prone'
    ]:
        data[feature_name] = data['CircuitName'].apply(
            lambda x: extract_circuit_features(x)[feature_name]
        )
    
    return data

def prepare_features_for_model(data, feature_cols):
    """Prepare features for model training or prediction."""
    if data is None:
        return None
    
    # Make sure all required feature columns exist
    for col in feature_cols[:]:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default values
    
    # Convert boolean features to int
    for col in feature_cols:
        if data[col].dtype == bool:
            data[col] = data[col].astype(int)
    
    # Handle missing values with median
    for col in feature_cols:
        if data[col].isna().any():
            if data[col].notna().any():
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = 0
    
    return data[feature_cols]