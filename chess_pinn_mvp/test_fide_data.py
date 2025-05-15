#!/usr/bin/env python3
"""
Test script for FIDE data extraction and storage.

This script:
1. Creates the necessary directory structure
2. Extracts data for a few grandmasters
3. Verifies that data is saved and can be loaded properly
"""
import os
import sys
import logging
from pathlib import Path

# Ensure data directories exist
from chess_pinn_mvp.utils.fide_data_manager import (
    get_player_rating_data, 
    RAW_DIR, 
    PROCESSED_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GM FIDE IDs
GM_IDS = {
    "Magnus Carlsen": "1503014",
    "Hikaru Nakamura": "2016192",
    "Gukesh D": "46616543",
    "Fabiano Caruana": "2020009",
    "Alireza Firouzja": "12573981",
    "Ding Liren": "8603677",
    "Wesley So": "5202213"
}

def main():
    """Main test function."""
    print("\n===== FIDE Data Extraction Test =====")
    
    # Ensure directory structure exists
    print("\nChecking data directories...")
    for directory in [RAW_DIR, PROCESSED_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")
    
    # Test data extraction for a few GMs
    for name, fide_id in GM_IDS.items():
        print(f"\nTesting data extraction for {name} (FIDE ID: {fide_id})")
        
        # First try loading from file
        print("Attempting to load from file (if exists)...")
        df = get_player_rating_data(fide_id, force_refresh=False)
        
        if df.empty:
            print("No data found or error loading. Fetching from FIDE website...")
            df = get_player_rating_data(fide_id, force_refresh=True)
        
        if df.empty:
            print(f"Failed to get data for {name}!")
            continue
        
        # Display basic info about the data
        print(f"Successfully retrieved {len(df)} rating points")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
        
        # Display a few samples
        print("\nSample data (first 3 entries):")
        print(df.head(3))
    
    print("\n===== Test Complete =====")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Processed data directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
