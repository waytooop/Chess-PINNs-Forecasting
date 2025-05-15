"""
FIDE data manager for chess rating data.

This module provides functionality to:
1. Fetch rating data from FIDE website
2. Store it in JSON files for easy access
3. Load data from files without re-scraping

This approach prioritizes reliability and simplicity.
"""
import os
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FIDE_PROFILE_BASE = "https://ratings.fide.com/profile"
FIDE_CHART_BASE = "https://ratings.fide.com/profile/{fide_id}/chart"

# Data directories
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_data_path(fide_id: str) -> Path:
    """Get path to raw data file."""
    return RAW_DIR / f"{fide_id}_ratings.json"


def get_processed_data_path(fide_id: str) -> Path:
    """Get path to processed data file."""
    return PROCESSED_DIR / f"{fide_id}_ratings.csv"


def save_raw_data(fide_id: str, data: Dict) -> None:
    """Save raw data to file."""
    path = get_raw_data_path(fide_id)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved raw data to {path}")


def load_raw_data(fide_id: str) -> Optional[Dict]:
    """Load raw data from file if it exists."""
    path = get_raw_data_path(fide_id)
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded raw data from {path}")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading raw data file: {e}")
    return None


def save_processed_data(fide_id: str, df: pd.DataFrame) -> None:
    """Save processed data to CSV file."""
    path = get_processed_data_path(fide_id)
    df.to_csv(path)
    logger.info(f"Saved processed data to {path}")


def load_processed_data(fide_id: str) -> Optional[pd.DataFrame]:
    """Load processed data from CSV file if it exists."""
    path = get_processed_data_path(fide_id)
    if path.exists():
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info(f"Loaded processed data from {path}")
            return df
        except Exception as e:
            logger.warning(f"Error reading processed data file: {e}")
    return None


def fetch_rating_history(fide_id: str, force_refresh: bool = False) -> List[Dict]:
    """
    Fetch rating history data from FIDE or load from file.
    
    Args:
        fide_id: FIDE ID to fetch data for
        force_refresh: Whether to force refresh data from FIDE website
        
    Returns:
        List of rating history data points
    """
    # Check if we have data saved
    if not force_refresh:
        raw_data = load_raw_data(fide_id)
        if raw_data and 'rating_history' in raw_data:
            return raw_data['rating_history']
    
    # Fetch from FIDE website
    logger.info(f"Fetching rating history for FIDE ID {fide_id} from website")
    
    # Generate URL
    url = FIDE_CHART_BASE.format(fide_id=fide_id)
    
    try:
        # Add a User-Agent to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Fetch page
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find rating history in JavaScript data
        chart_data = []
        scripts = soup.find_all("script")
        
        for script in scripts:
            if script.string and "chart.data.setColumns" in script.string:
                # Extract data using string manipulation (more reliable than regex)
                data_str = script.string
                
                # Find start of data array
                start_idx = data_str.find("chart.data.setColumns(")
                if start_idx == -1:
                    continue
                
                # Find brackets
                open_bracket = data_str.find("[", start_idx)
                close_bracket = data_str.rfind("]", open_bracket)
                
                if open_bracket == -1 or close_bracket == -1:
                    continue
                
                # Extract data array
                data_array = data_str[open_bracket:close_bracket+1]
                
                # Clean and parse data
                clean_data = data_array.replace("'", '"')
                
                try:
                    # Try to parse as JSON
                    parsed_data = json.loads(clean_data)
                    
                    # Process data
                    for item in parsed_data:
                        if isinstance(item, list) and len(item) == 2:
                            date_str, rating = item
                            
                            # Try to parse date (format: "YYYY-MM")
                            try:
                                year, month = map(int, date_str.split('-'))
                                date = datetime(year, month, 1)
                                
                                chart_data.append({
                                    "date": date.strftime("%Y-%m-%d"),
                                    "rating": int(rating)
                                })
                            except (ValueError, TypeError):
                                logger.warning(f"Failed to parse date: {date_str}")
                                continue
                except json.JSONDecodeError:
                    # If JSON parsing fails, try manual extraction with string splitting
                    parts = data_array.split("],[")
                    for part in parts:
                        clean_part = part.strip("[]")
                        items = clean_part.split(",")
                        
                        if len(items) >= 2:
                            date_str = items[0].strip("'\" ")
                            rating_str = items[1].strip()
                            
                            try:
                                year, month = map(int, date_str.split('-'))
                                date = datetime(year, month, 1)
                                rating = int(rating_str)
                                
                                chart_data.append({
                                    "date": date.strftime("%Y-%m-%d"),
                                    "rating": rating
                                })
                            except (ValueError, TypeError):
                                continue
        
        # Sort by date
        chart_data.sort(key=lambda x: x["date"])
        
        # Save raw data
        raw_data = {
            "fide_id": fide_id,
            "fetch_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rating_history": chart_data
        }
        save_raw_data(fide_id, raw_data)
        
        return chart_data
        
    except Exception as e:
        logger.error(f"Error fetching rating history: {e}")
        
        # Try to load from cache as a fallback
        raw_data = load_raw_data(fide_id)
        if raw_data and 'rating_history' in raw_data:
            logger.info("Using cached data as fallback")
            return raw_data['rating_history']
        
        # If all else fails, return empty list
        return []


def process_rating_data(fide_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Process rating data for a player.
    
    Args:
        fide_id: FIDE ID to process
        force_refresh: Whether to force refresh data
        
    Returns:
        DataFrame with processed rating data
    """
    # Check if we have processed data saved
    if not force_refresh:
        df = load_processed_data(fide_id)
        if df is not None and not df.empty:
            return df
    
    # Fetch raw data
    rating_history = fetch_rating_history(fide_id, force_refresh)
    
    if not rating_history:
        logger.warning(f"No rating history found for FIDE ID {fide_id}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(rating_history)
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Basic preprocessing
    # Sort by date
    df = df.sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Ensure we have enough data
    if len(df) < 5:
        logger.warning(f"Insufficient data points for {fide_id}: {len(df)}")
        return pd.DataFrame()
    
    # Add derived features
    # Normalize time (days since start)
    start_date = df.index.min()
    df['t'] = (df.index - start_date).days
    
    # Save processed data
    save_processed_data(fide_id, df)
    
    return df


def get_player_rating_data(fide_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Main function to get rating data for a player.
    
    Args:
        fide_id: FIDE ID to get data for
        force_refresh: Whether to force refresh data
        
    Returns:
        DataFrame with processed rating data
    """
    return process_rating_data(fide_id, force_refresh)


# Provide an example function for testing
def test_data_extraction():
    """Test data extraction for Magnus Carlsen."""
    fide_id = "1503014"  # Magnus Carlsen
    
    print(f"Testing data extraction for FIDE ID {fide_id} (Magnus Carlsen)")
    
    # Get data
    df = get_player_rating_data(fide_id)
    
    if df.empty:
        print("Failed to get data!")
        return
    
    print(f"Successfully retrieved {len(df)} rating points")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
    
    print("\nFirst 5 entries:")
    print(df.head())
    
    print(f"\nData saved to: {get_processed_data_path(fide_id)}")


if __name__ == "__main__":
    test_data_extraction()
