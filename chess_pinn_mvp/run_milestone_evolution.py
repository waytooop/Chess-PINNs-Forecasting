#!/usr/bin/env python3
"""
Script to generate chess rating milestone evolution visualizations.

This script creates visualizations showing how milestone predictions evolve
throughout a player's career as more data becomes available.
"""
import os
import json
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from chess_pinn_mvp.visualization.career_milestone_visualizer import visualize_career_milestone_evolution
from chess_pinn_mvp.utils.real_data_loader import load_tournament_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_milestone_evolution_analysis(
    fide_id: str,
    milestone_counts: int = 5,
    prediction_points_count: int = 6,
    model_type: str = 'nonlinear',
    forecast_years: int = 10,
    epochs: int = 200,
    output_dir: str = None
):
    """
    Run milestone evolution analysis for a specific player.
    
    Args:
        fide_id: FIDE ID of the player
        milestone_counts: Number of milestone levels to predict
        prediction_points_count: Number of points in career to make predictions from
        model_type: Type of model to use ('linear' or 'nonlinear')
        forecast_years: Number of years to forecast from each prediction point
        epochs: Number of training epochs for each model
        output_dir: Directory to save output (default is chess_pinn_mvp/output/figures)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load player data
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           'data', 'raw', f"{fide_id}_ratings.json")
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    with open(data_file, 'r') as f:
        player_data = json.load(f)
    
    # Extract rating history
    rating_history = player_data.get('rating_history', [])
    
    if not rating_history:
        logger.error(f"No rating history found for player {fide_id}")
        return
    
    # Get player name from the tournament data for better titles
    player_name = None
    tournament_data = load_tournament_data()
    for player in tournament_data.get('players', []):
        if player.get('fide_id') == fide_id:
            player_name = player.get('name')
            break
    
    if player_name is None:
        player_name = f"Player {fide_id}"
    
    # Convert rating history to proper format for visualization
    dates = [entry['date'] for entry in rating_history]
    ratings = [entry['rating'] for entry in rating_history]
    
    # Analyze rating range to determine milestone levels
    min_rating = min(ratings)
    max_rating = max(ratings)
    rating_range = max_rating - min_rating
    
    # Define milestone levels (evenly spaced)
    if milestone_counts > 1:
        milestone_step = rating_range / (milestone_counts - 1)
        base_milestones = [
            int(round(min_rating + i * milestone_step))
            for i in range(milestone_counts)
        ]
    else:
        base_milestones = [max_rating]
    
    # Add some higher milestones beyond current maximum
    extra_milestones = [max_rating + 100 * i for i in range(1, 3)]
    milestones = sorted(set(base_milestones + extra_milestones))
    
    # Ensure milestones are divisible by 50 or 100 for cleaner visualization
    milestones = [int(round(m / 50) * 50) for m in milestones]
    
    # Define prediction points (evenly spaced in the dataset)
    total_points = len(rating_history)
    if total_points < prediction_points_count:
        prediction_points = list(range(total_points))
    else:
        step = total_points // (prediction_points_count - 1)
        prediction_points = [i * step for i in range(prediction_points_count - 1)] + [total_points - 1]
    
    # Log analysis parameters
    logger.info(f"Running milestone evolution analysis for {player_name} (FIDE ID: {fide_id})")
    logger.info(f"Rating range: {min_rating} to {max_rating}")
    logger.info(f"Selected milestones: {milestones}")
    logger.info(f"Prediction points: {[rating_history[i]['date'] for i in prediction_points]}")
    
    # Create output path
    output_path = os.path.join(output_dir, f"{fide_id}_{model_type}_milestone_evolution.png")
    
    # Run visualization
    visualize_career_milestone_evolution(
        rating_history=rating_history,
        prediction_points=prediction_points,
        milestones=milestones,
        output_path=output_path,
        model_type=model_type,
        forecast_years=forecast_years,
        confidence_bands=True,
        title=f"{player_name}: Rating Milestone Evolution ({model_type.capitalize()} Model)",
        epochs=epochs
    )
    
    logger.info(f"Milestone evolution visualization saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate chess rating milestone evolution visualizations")
    parser.add_argument('--fide_id', type=str, default="1503014", help="FIDE ID of the player to analyze")
    parser.add_argument('--model', type=str, default="nonlinear", choices=["linear", "nonlinear"], 
                        help="Type of model to use (linear or nonlinear)")
    parser.add_argument('--milestones', type=int, default=5, help="Number of milestone levels to predict")
    parser.add_argument('--predictions', type=int, default=6, help="Number of prediction points in career")
    parser.add_argument('--forecast_years', type=int, default=10, help="Number of years to forecast from each point")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs for each model")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save output")
    
    args = parser.parse_args()
    
    run_milestone_evolution_analysis(
        fide_id=args.fide_id,
        milestone_counts=args.milestones,
        prediction_points_count=args.predictions,
        model_type=args.model,
        forecast_years=args.forecast_years,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
