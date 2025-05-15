"""
Data processing utilities for chess rating data.

This module handles:
1. Loading player rating data from FIDE
2. Preprocessing and normalization
3. Preparing training data for PINN models
4. Creating PyTorch datasets for model training
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

from chess_pinn_mvp.utils.data_manager import get_player_rating_data
from chess_pinn_mvp.utils.gm_config import get_gm_by_fide_id, get_gm_by_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RatingDataset(Dataset):
    """Dataset for training the PINN model on rating time series."""
    
    def __init__(self, data: Dict, normalize: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data: Dictionary with keys 't', 'rating', and 'mask'
            normalize: Whether to normalize data
        """
        self.t = torch.tensor(data['t'], dtype=torch.float32)
        self.rating = torch.tensor(data['rating'], dtype=torch.float32)
        self.mask = torch.tensor(data['mask'], dtype=torch.bool)
        
        # Get valid data points (where mask is True)
        self.valid_indices = torch.nonzero(self.mask.view(-1)).squeeze()
        
        # Normalize data if requested
        if normalize:
            # Normalize ratings to have mean 0 and std 1
            self.rating_mean = self.rating[self.mask].mean().item()
            self.rating_std = self.rating[self.mask].std().item()
            self.rating_normalized = (self.rating - self.rating_mean) / self.rating_std
        else:
            self.rating_normalized = self.rating
            self.rating_mean = 0.0
            self.rating_std = 1.0
    
    def __len__(self):
        """Get number of valid data points."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a data point by index."""
        i = self.valid_indices[idx]
        
        return {
            't': self.t[i],
            'r': self.rating_normalized[i],
            'r_original': self.rating[i]
        }
    
    def get_all_valid_data(self):
        """Get all valid data points."""
        return {
            't': self.t[self.mask],
            'r': self.rating_normalized[self.mask],
            'r_original': self.rating[self.mask]
        }
    
    def get_normalization_params(self):
        """Get normalization parameters."""
        return {
            'rating_mean': self.rating_mean,
            'rating_std': self.rating_std
        }
    
    def denormalize_rating(self, normalized_rating):
        """Convert normalized rating back to original scale."""
        return normalized_rating * self.rating_std + self.rating_mean


def prepare_training_data(
    rating_data: pd.DataFrame,
    test_fraction: float = 0.2,
    validate_data: bool = True
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Prepare training and validation data from a DataFrame.
    
    Args:
        rating_data: DataFrame with rating data (must have 'rating' and 't' columns)
        test_fraction: Fraction of data to use for validation
        validate_data: Whether to validate data (check for sufficient points, etc.)
        
    Returns:
        Tuple of (train_data, test_data), or (train_data, None) if test_fraction is 0,
        or (None, None) if data preparation fails
    """
    if rating_data.empty:
        logger.warning("Empty DataFrame provided")
        return None, None
    
    # Ensure we have the required columns
    if 'rating' not in rating_data.columns or 't' not in rating_data.columns:
        logger.warning("DataFrame must have 'rating' and 't' columns")
        return None, None
    
    # Log basic info
    logger.info(f"Processing rating data")
    logger.info(f"Date range: {rating_data.index.min()} to {rating_data.index.max()}")
    logger.info(f"Rating range: {rating_data['rating'].min()} to {rating_data['rating'].max()}")
    logger.info(f"Number of data points: {len(rating_data)}")
    
    # Basic validation
    if validate_data and len(rating_data) < 10:
        logger.warning(f"Insufficient data points: {len(rating_data)}")
        return None, None
    
    # Sort by date
    df = rating_data.sort_index()
    
    # Get values
    t_values = df['t'].values
    r_values = df['rating'].values
    
    # Create mask (all data points are valid in this case)
    mask = np.ones_like(t_values, dtype=bool)
    
    # Create data dictionary
    data = {
        't': t_values,
        'rating': r_values,
        'mask': mask
    }
    
    # Split into training and validation sets
    if test_fraction > 0:
        # Use time-based split (early data for training, late data for testing)
        split_idx = int((1 - test_fraction) * len(df))
        
        train_data = {
            't': data['t'][:split_idx],
            'rating': data['rating'][:split_idx],
            'mask': data['mask'][:split_idx]
        }
        
        test_data = {
            't': data['t'][split_idx:],
            'rating': data['rating'][split_idx:],
            'mask': data['mask'][split_idx:]
        }
        
        return train_data, test_data
    else:
        # No validation set
        return data, None


def prepare_training_data_for_player(
    fide_id: str,
    test_fraction: float = 0.2,
    force_refresh: bool = False,
    validate_data: bool = True
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Prepare training and validation data for a player.
    
    Args:
        fide_id: FIDE ID of the player
        test_fraction: Fraction of data to use for validation
        force_refresh: Whether to force refresh data from FIDE
        validate_data: Whether to validate data (check for sufficient points, etc.)
        
    Returns:
        Tuple of (train_data, test_data), or (train_data, None) if test_fraction is 0,
        or (None, None) if data preparation fails
    """
    # Get player data
    df = get_player_rating_data(fide_id, force_refresh=force_refresh)
    
    if df.empty:
        logger.warning(f"No data found for player with FIDE ID {fide_id}")
        return None, None
    
    # Log basic info
    logger.info(f"Processing data for player with FIDE ID {fide_id}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
    logger.info(f"Number of data points: {len(df)}")
    
    # Basic validation
    if validate_data and len(df) < 10:
        logger.warning(f"Insufficient data points for FIDE ID {fide_id}: {len(df)}")
        return None, None
    
    # Sort by date
    df = df.sort_index()
    
    # Create time axis (days since start)
    t_values = df['t'].values
    r_values = df['rating'].values
    
    # Create mask (all data points are valid in this case)
    mask = np.ones_like(t_values, dtype=bool)
    
    # Create data dictionary
    data = {
        't': t_values,
        'rating': r_values,
        'mask': mask
    }
    
    # Split into training and validation sets
    if test_fraction > 0:
        # Use time-based split (early data for training, late data for testing)
        split_idx = int((1 - test_fraction) * len(df))
        
        train_data = {
            't': data['t'][:split_idx],
            'rating': data['rating'][:split_idx],
            'mask': data['mask'][:split_idx]
        }
        
        test_data = {
            't': data['t'][split_idx:],
            'rating': data['rating'][split_idx:],
            'mask': data['mask'][split_idx:]
        }
        
        return train_data, test_data
    else:
        # No validation set
        return data, None


def create_data_loaders(
    train_dataset: RatingDataset, 
    val_dataset: Optional[RatingDataset] = None,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create DataLoader instances for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader), or (train_loader, None) if val_dataset is None
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return train_loader, val_loader
    else:
        return train_loader, None


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move batch to device."""
    return {k: v.to(device) for k, v in batch.items()}


def load_and_process_rating_data(
    fide_id: str,
    force_refresh: bool = False
) -> Optional[pd.DataFrame]:
    """
    Load and process rating data for a player with the given FIDE ID.
    
    Args:
        fide_id: FIDE ID of the player
        force_refresh: Whether to force refresh data from FIDE
        
    Returns:
        DataFrame with processed rating data, or None if data loading fails
    """
    # Get player data
    df = get_player_rating_data(fide_id, force_refresh=force_refresh)
    
    if df.empty:
        logger.warning(f"No data found for player with FIDE ID {fide_id}")
        return None
    
    # Log basic info
    logger.info(f"Processing data for player with FIDE ID {fide_id}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
    logger.info(f"Number of data points: {len(df)}")
    
    # Basic validation
    if len(df) < 10:
        logger.warning(f"Insufficient data points for FIDE ID {fide_id}: {len(df)}")
        return None
    
    # Sort by date
    df = df.sort_index()
    
    return df


def prepare_gm_data(
    gm_key: str,
    test_fraction: float = 0.2,
    force_refresh: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict]]:
    """
    Prepare data for a grandmaster.
    
    Args:
        gm_key: Grandmaster key (e.g., 'GM001')
        test_fraction: Fraction of data to use for validation
        force_refresh: Whether to force refresh data from FIDE
        
    Returns:
        Tuple of (df, train_data, test_data), or (None, None, None) if data preparation fails
    """
    # Get GM info
    gm_info = get_gm_by_key(gm_key)
    
    if not gm_info:
        logger.error(f"No GM found with key {gm_key}")
        return None, None, None
    
    fide_id = gm_info['id']
    
    # Get player data
    df = get_player_rating_data(fide_id, force_refresh=force_refresh)
    
    if df.empty:
        logger.warning(f"No data found for {gm_info['name']} (FIDE ID: {fide_id})")
        return None, None, None
    
    # Prepare training data
    train_data, test_data = prepare_training_data_for_player(
        fide_id=fide_id,
        test_fraction=test_fraction,
        force_refresh=force_refresh
    )
    
    if train_data is None:
        logger.warning(f"Failed to prepare training data for {gm_info['name']}")
        return df, None, None
    
    return df, train_data, test_data
