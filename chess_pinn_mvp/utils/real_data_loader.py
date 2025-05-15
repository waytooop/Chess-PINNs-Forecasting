"""
Real FIDE tournament data loader for chess rating analysis.

This utility module helps load and format real FIDE tournament data
that was manually collected and formatted according to the 
real_tournament_history_schema.json structure.

This module works with actual historical FIDE rating data for
accurate and realistic chess rating prediction models.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
REAL_DATA_FILE = os.path.join(DATA_DIR, 'real_tournament_history.json')
SCHEMA_FILE = os.path.join(DATA_DIR, 'real_tournament_history_schema.json')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


def load_schema() -> Dict[str, Any]:
    """
    Load the tournament history schema file.
    
    Returns:
        Dictionary containing the schema
    """
    if not os.path.exists(SCHEMA_FILE):
        logger.error(f"Schema file not found: {SCHEMA_FILE}")
        return {}
    
    with open(SCHEMA_FILE, 'r') as f:
        return json.load(f)


def save_tournament_data(data: Dict[str, Any]) -> bool:
    """
    Save tournament data to the real tournament history file.
    
    Args:
        data: Dictionary containing tournament data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(REAL_DATA_FILE), exist_ok=True)
        
        # Save data
        with open(REAL_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Tournament data saved to {REAL_DATA_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save tournament data: {e}")
        return False


def load_tournament_data() -> Dict[str, Any]:
    """
    Load the tournament history data file if it exists,
    otherwise load the schema as a starting point.
    
    Returns:
        Dictionary containing tournament data
    """
    if os.path.exists(REAL_DATA_FILE):
        try:
            with open(REAL_DATA_FILE, 'r') as f:
                data = json.load(f)
            logger.info(f"Tournament data loaded from {REAL_DATA_FILE}")
            return data
        except Exception as e:
            logger.error(f"Failed to load tournament data: {e}")
            return load_schema()
    else:
        logger.info(f"Tournament data file not found, loading schema")
        return load_schema()


def add_player_ratings(
    fide_id: str,
    ratings_data: List[Dict[str, Any]],
    save: bool = True
) -> bool:
    """
    Add or update rating data for a player.
    
    Args:
        fide_id: FIDE ID of the player
        ratings_data: List of rating data dictionaries, each containing
                      at least 'date' and 'rating' keys
        save: Whether to save the updated data to file
        
    Returns:
        True if successful, False otherwise
    """
    # Load existing data
    data = load_tournament_data()
    
    # Find player by FIDE ID
    player_found = False
    for player in data.get('players', []):
        if player.get('fide_id') == fide_id:
            # Update player's ratings
            player['standard_ratings'] = ratings_data
            player_found = True
            break
    
    if not player_found:
        logger.error(f"Player with FIDE ID {fide_id} not found in schema")
        return False
    
    # Save updated data if requested
    if save:
        return save_tournament_data(data)
    
    return True


def convert_to_processed_data(fide_id: str) -> bool:
    """
    Convert real tournament data for a player to processed CSV format
    compatible with the PINN models.
    
    Args:
        fide_id: FIDE ID of the player
        
    Returns:
        True if successful, False otherwise
    """
    # Load tournament data
    data = load_tournament_data()
    
    # Find player
    player_data = None
    for player in data.get('players', []):
        if player.get('fide_id') == fide_id:
            player_data = player
            break
    
    if player_data is None:
        logger.error(f"Player with FIDE ID {fide_id} not found")
        return False
    
    # Check if player has ratings
    ratings = player_data.get('standard_ratings', [])
    if not ratings:
        logger.error(f"No rating data found for player {fide_id}")
        return False
    
    # Convert to DataFrame
    df_data = []
    for entry in ratings:
        try:
            date = datetime.strptime(entry['date'], '%Y-%m-%d')
            rating = int(entry['rating'])
            df_data.append({'date': date, 'rating': rating})
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid rating entry: {entry}, error: {e}")
    
    if not df_data:
        logger.error(f"No valid rating data found for player {fide_id}")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    
    # Add time since start in days
    df['t'] = (df.index - df.index.min()).days
    
    # Save to processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, f"{fide_id}_ratings.csv")
    df.to_csv(output_path)
    
    logger.info(f"Processed data for {fide_id} saved to {output_path}")
    return True


def load_real_player_data(fide_id: str) -> pd.DataFrame:
    """
    Load real player data for use with PINN models.
    Converts from tournament history data if needed.
    
    Args:
        fide_id: FIDE ID of the player
        
    Returns:
        DataFrame with player rating data
    """
    # First check if processed data exists
    processed_path = os.path.join(PROCESSED_DIR, f"{fide_id}_ratings.csv")
    
    if os.path.exists(processed_path):
        # Load existing data
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded processed real data from {processed_path}")
        return df
    
    # If not, try to convert from real tournament data
    if convert_to_processed_data(fide_id):
        # Now load the newly created file
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        logger.info(f"Converted and loaded real data for {fide_id}")
        return df
    
    # If all else fails, return empty DataFrame
    logger.error(f"Failed to load real data for {fide_id}")
    return pd.DataFrame()


def get_player_names() -> List[Dict[str, str]]:
    """
    Get a list of players from the tournament data.
    
    Returns:
        List of dictionaries with player names and FIDE IDs
    """
    data = load_tournament_data()
    return [
        {'name': player.get('name', ''), 'fide_id': player.get('fide_id', '')}
        for player in data.get('players', [])
    ]


def has_rating_data(fide_id: str) -> bool:
    """
    Check if a player has rating data.
    
    Args:
        fide_id: FIDE ID of the player
        
    Returns:
        True if the player has rating data, False otherwise
    """
    data = load_tournament_data()
    
    for player in data.get('players', []):
        if player.get('fide_id') == fide_id:
            return bool(player.get('standard_ratings', []))
    
    return False


if __name__ == "__main__":
    # Example usage
    print("Real FIDE Tournament Data Utility")
    print("=================================")
    
    # Load schema
    schema = load_schema()
    print(f"Found {len(schema.get('players', []))} players in schema")
    
    # Check if data file exists
    if os.path.exists(REAL_DATA_FILE):
        print(f"Real tournament data file exists: {REAL_DATA_FILE}")
        data = load_tournament_data()
        
        # Print player stats
        for player in data.get('players', []):
            name = player.get('name', 'Unknown')
            fide_id = player.get('fide_id', 'Unknown')
            ratings = player.get('standard_ratings', [])
            
            if ratings:
                print(f"{name} (FIDE ID: {fide_id}): {len(ratings)} rating entries")
            else:
                print(f"{name} (FIDE ID: {fide_id}): No rating data")
    else:
        print(f"Real tournament data file does not exist yet")
        print(f"Use the add_player_ratings() function to add data")
    
    # Example of how to convert data
    # convert_to_processed_data("1503014")  # Magnus Carlsen
