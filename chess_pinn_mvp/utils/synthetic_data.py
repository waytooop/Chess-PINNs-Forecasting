"""
Synthetic data generator for chess rating data.

This module generates realistic-looking rating history data for demonstration purposes,
when actual FIDE data cannot be accessed.

Note: This is only for demonstration purposes. In a production environment,
you would use actual historical data from FIDE or other chess rating databases.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from chess_pinn_mvp.utils.gm_config import GRANDMASTERS, GM_DICT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directories
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')


class SyntheticRatingGenerator:
    """Generator for synthetic rating history data."""
    
    @staticmethod
    def generate_data_for_gm(
        gm_key: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'M',  # Monthly by default
        noise_level: float = 0.5,
        trend_strength: float = 0.7,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic rating history for a grandmaster.
        
        Args:
            gm_key: Grandmaster key (e.g., 'GM001')
            start_date: Start date for the data (default: 10 years ago)
            end_date: End date for the data (default: now)
            frequency: Sampling frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            noise_level: Level of random noise (0-1)
            trend_strength: Strength of upward trend for young players (0-1)
            save: Whether to save the data to files
            
        Returns:
            DataFrame with synthetic rating data
        """
        # Get GM info
        gm_info = GM_DICT.get(gm_key)
        if not gm_info:
            logger.error(f"GM with key {gm_key} not found")
            return pd.DataFrame()
        
        fide_id = gm_info['id']
        name = gm_info['name']
        born_year = gm_info['born']
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            # Start 10 years ago or when GM was 10 years old, whichever is later
            years_ago = 10
            gm_at_10 = datetime(born_year + 10, 1, 1)
            ten_years_ago = end_date - timedelta(days=365 * years_ago)
            start_date = max(gm_at_10, ten_years_ago)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Calculate GM's age throughout the date range
        ages = [(date.year - born_year) + (date.month / 12) for date in date_range]
        
        # Generate base ratings based on age and experience
        # Start with a reasonable initial rating and apply growth curve
        current_age = (end_date.year - born_year) + (end_date.month / 12)
        
        # Different parameters based on GM's current strength and age
        if current_age < 25:
            # Young and rising GM
            initial_rating = 2200
            max_rating = 2800 + np.random.normal(0, 50)
            growth_factor = 0.4  # Faster growth
        else:
            # Established GM
            initial_rating = 2400
            max_rating = 2750 + np.random.normal(0, 100)
            growth_factor = 0.2  # Slower growth
        
        # Create growth curve
        ratings = []
        current_rating = initial_rating
        
        for i, age in enumerate(ages):
            # Calculate development factor (diminishes with age)
            development_factor = np.maximum(0, 1 - (age / 40))
            
            # Calculate trend (stronger for younger players)
            if i > 0:
                trend = trend_strength * development_factor * (max_rating - current_rating) / 500
                
                # Add noise (normalized by player level)
                noise = np.random.normal(0, noise_level * 20 * (1 - development_factor * 0.5))
                
                # Update current rating
                current_rating = current_rating + trend + noise
                
                # Ensure rating stays within reasonable bounds
                current_rating = np.clip(current_rating, initial_rating, max_rating)
            
            ratings.append(int(current_rating))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'rating': ratings
        })
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Add time since start in days
        df['t'] = (df.index - df.index.min()).days
        
        # Log creation
        logger.info(f"Generated synthetic data for {name} with {len(df)} data points")
        
        # Save data if requested
        if save:
            # Save raw data in JSON format
            raw_data = {
                "fide_id": fide_id,
                "fetch_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_synthetic": True,
                "rating_history": [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "rating": rating
                    }
                    for date, rating in zip(df.index, df['rating'])
                ]
            }
            
            raw_path = os.path.join(RAW_DIR, f"{fide_id}_ratings.json")
            with open(raw_path, 'w') as f:
                json.dump(raw_data, f, indent=2)
            logger.info(f"Saved synthetic raw data to {raw_path}")
            
            # Save processed data as CSV
            processed_path = os.path.join(PROCESSED_DIR, f"{fide_id}_ratings.csv")
            df.to_csv(processed_path)
            logger.info(f"Saved synthetic processed data to {processed_path}")
        
        return df
    
    @staticmethod
    def generate_data_for_all_gms(save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for all grandmasters in the configuration.
        
        Args:
            save: Whether to save the data to files
            
        Returns:
            Dictionary mapping GM keys to their synthetic data
        """
        result = {}
        
        for gm in GRANDMASTERS:
            key = gm['key']
            df = SyntheticRatingGenerator.generate_data_for_gm(key, save=save)
            result[key] = df
        
        return result


def get_synthetic_player_data(fide_id: str) -> Dict[str, Any]:
    """
    Get synthetic player data for a FIDE ID.
    
    Args:
        fide_id: FIDE ID to get data for
        
    Returns:
        Dictionary with player data including rating history
    """
    # Check if we already have raw data
    raw_path = os.path.join(RAW_DIR, f"{fide_id}_ratings.json")
    
    if os.path.exists(raw_path):
        # Load existing data
        with open(raw_path, 'r') as f:
            return json.load(f)
    
    # Find GM info from FIDE ID
    gm_info = None
    for gm in GRANDMASTERS:
        if gm['id'] == fide_id:
            gm_info = gm
            break
    
    if not gm_info:
        logger.error(f"GM with FIDE ID {fide_id} not found")
        return {}
    
    # Generate synthetic data
    SyntheticRatingGenerator.generate_data_for_gm(gm_info['key'], save=True)
    
    # Load and return the newly generated data
    with open(raw_path, 'r') as f:
        return json.load(f)


def load_or_generate_data(fide_id: str) -> pd.DataFrame:
    """
    Load existing or generate new synthetic data for a player.
    
    Args:
        fide_id: FIDE ID to get data for
        
    Returns:
        DataFrame with rating data
    """
    # Check if we already have processed data
    processed_path = os.path.join(PROCESSED_DIR, f"{fide_id}_ratings.csv")
    
    if os.path.exists(processed_path):
        # Load existing data
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded synthetic processed data from {processed_path}")
        return df
    
    # Find GM info from FIDE ID
    gm_info = None
    for gm in GRANDMASTERS:
        if gm['id'] == fide_id:
            gm_info = gm
            break
    
    if not gm_info:
        logger.error(f"GM with FIDE ID {fide_id} not found")
        return pd.DataFrame()
    
    # Generate synthetic data
    return SyntheticRatingGenerator.generate_data_for_gm(gm_info['key'], save=True)


if __name__ == "__main__":
    # Generate data for all GMs
    print("Generating synthetic rating data for all grandmasters...")
    gm_data = SyntheticRatingGenerator.generate_data_for_all_gms(save=True)
    
    # Print some statistics
    print("\nGenerated data summary:")
    for key, df in gm_data.items():
        gm_info = GM_DICT[key]
        print(f"{gm_info['name']} ({key}):")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Rating range: {df['rating'].min()} to {df['rating'].max()}")
        print(f"  Data points: {len(df)}")
        print()
