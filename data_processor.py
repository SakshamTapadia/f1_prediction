import numpy as np
import pandas as pd
from config import logger

def preprocess_race_data(session):
    """Extract relevant features from the race session with enhanced circuit data."""
    try:
        laps = session.laps
        
        if laps.empty:
            logger.warning(f"No lap data available for {session.event.year} {session.event.name}")
            return None
        
        # Add race metadata
        laps['Year'] = session.event.year
        laps['CircuitName'] = session.event.name
        laps['CircuitShortName'] = session.event['EventName']
        
        # Extract weather data
        if hasattr(session, 'weather_data') and not session.weather_data.empty:
            weather = session.weather_data
            avg_temp = weather['AirTemp'].mean()
            avg_track_temp = weather['TrackTemp'].mean()
            avg_humidity = weather['Humidity'].mean()
            laps['AirTemp'] = avg_temp
            laps['TrackTemp'] = avg_track_temp
            laps['Humidity'] = avg_humidity
        else:
            # Use default values if weather data is missing
            laps['AirTemp'] = np.nan
            laps['TrackTemp'] = np.nan
            laps['Humidity'] = np.nan
        
        # Convert lap times to numeric format
        valid_laps = laps.dropna(subset=['LapTime'])
        if valid_laps.empty:
            logger.warning(f"No valid lap times for {session.event.year} {session.event.name}")
            return None
            
        valid_laps['LapTime (s)'] = valid_laps['LapTime'].dt.total_seconds()
        
        # Add track position and stint information
        valid_laps['TrackPosition'] = valid_laps['Position']
        valid_laps['Stint'] = valid_laps['Stint']
        
        # Add tire data if available
        if 'Compound' in valid_laps.columns:
            valid_laps['TireCompound'] = valid_laps['Compound']
        
        # Add driver information
        if hasattr(session, 'results') and not session.results.empty:
            drivers = session.results[['DriverNumber', 'Abbreviation', 'FullName', 'TeamName']]
            merged_data = pd.merge(valid_laps, drivers, left_on="DriverNumber", right_on="DriverNumber", how="left")
            
            # Calculate average lap time per driver
            avg_laptimes = merged_data.groupby(['FullName', 'TeamName', 'Year', 'CircuitName'])[
                'LapTime (s)', 'AirTemp', 'TrackTemp', 'Humidity'
            ].agg({
                'LapTime (s)': ['mean', 'min', 'std'],
                'AirTemp': 'mean',
                'TrackTemp': 'mean',
                'Humidity': 'mean'
            }).reset_index()
            
            # Flatten multi-level columns
            avg_laptimes.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in avg_laptimes.columns]
            
            return avg_laptimes
        else:
            logger.warning(f"No driver results available for {session.event.year} {session.event.name}")
            return None
    
    except Exception as e:
        logger.error(f"Error processing race data: {e}")
        return None

def preprocess_quali_data(session):
    """Extract qualifying data with enhanced circuit information."""
    try:
        if hasattr(session, 'results') and not session.results.empty:
            quali = session.results.copy()
            
            # Add circuit information
            quali['Year'] = session.event.year
            quali['CircuitName'] = session.event.name
            quali['CircuitShortName'] = session.event['EventName']
            
            # Extract weather data
            if hasattr(session, 'weather_data') and not session.weather_data.empty:
                weather = session.weather_data
                avg_temp = weather['AirTemp'].mean()
                avg_track_temp = weather['TrackTemp'].mean()
                avg_humidity = weather['Humidity'].mean()
                quali['AirTemp'] = avg_temp
                quali['TrackTemp'] = avg_track_temp
                quali['Humidity'] = avg_humidity
            else:
                # Use default values if weather data is missing
                quali['AirTemp'] = np.nan
                quali['TrackTemp'] = np.nan
                quali['Humidity'] = np.nan
            
            # Convert qualifying times to numeric format
            for col in ['Q1', 'Q2', 'Q3']:
                if col in quali.columns:
                    # Handle NaN values safely
                    mask = quali[col].notna()
                    if mask.any():
                        quali.loc[mask, col] = pd.to_timedelta(quali.loc[mask, col]).dt.total_seconds()
            
            # Calculate the best qualifying time
            best_q_columns = [col for col in ['Q1', 'Q2', 'Q3'] if col in quali.columns]
            if best_q_columns:
                # Only use non-NaN values for minimum calculation
                quali['BestQualiTime'] = quali[best_q_columns].min(axis=1)
            
            return quali
        else:
            logger.warning(f"No qualifying results for {session.event.year} {session.event.name}")
            return None
    
    except Exception as e:
        logger.error(f"Error processing qualifying data: {e}")
        return None