"""
Data management utilities for chess rating data.

This module handles loading, caching, and processing chess rating data.
It uses synthetic data generated from the synthetic_data module when actual FIDE data
cannot be accessed.
"""
import os
import json
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from chess_pinn_mvp.utils.fide_data_manager import get_player_rating_data as get_fide_data
from chess_pinn_mvp.utils.synthetic_data import load_or_generate_data, get_synthetic_player_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_player_data(fide_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get player data, trying FIDE first, then falling back to synthetic data.
    
    Args:
        fide_id: FIDE ID of the player
        force_refresh: Whether to force refresh data
        
    Returns:
        Dictionary with player data
    """
    try:
        # First try to get data from FIDE
        fide_data = get_fide_data(fide_id, force_refresh=force_refresh)
        
        # Check if rating history exists and has data
        if fide_data and fide_data.get('rating_history') and len(fide_data.get('rating_history', [])) > 0:
            logger.info(f"Successfully fetched FIDE data for {fide_id}")
            return fide_data
    except Exception as e:
        logger.warning(f"Failed to fetch FIDE data: {e}")
    
    # Fall back to synthetic data
    logger.info(f"Using synthetic data for {fide_id}")
    return get_synthetic_player_data(fide_id)


def get_player_rating_data(fide_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Get processed rating data for a player.
    
    Args:
        fide_id: FIDE ID of the player
        force_refresh: Whether to force refresh data
        
    Returns:
        DataFrame with processed rating data
    """
    try:
        # First try to get processed data from FIDE
        df = get_fide_data(fide_id, force_refresh=force_refresh)
        
        if not df.empty:
            logger.info(f"Successfully loaded FIDE processed data for {fide_id}")
            return df
    except Exception as e:
        logger.warning(f"Failed to load FIDE processed data: {e}")
    
    # Fall back to synthetic data
    logger.info(f"Using synthetic data for {fide_id}")
    return load_or_generate_data(fide_id)


def process_player(fide_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Process rating data for a player.
    
    Args:
        fide_id: FIDE ID of the player
        force_refresh: Whether to force refresh data
        
    Returns:
        DataFrame with processed rating data
    """
    return get_player_rating_data(fide_id, force_refresh=force_refresh)
