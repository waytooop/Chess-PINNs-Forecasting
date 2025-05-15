#!/usr/bin/env python3
"""
Main application for Chess-PINNs-Forecasting MVP.

This script:
1. Loads synthetic rating data for multiple grandmasters
2. Trains both linear and nonlinear PINN models for each GM
3. Creates comparative visualizations between GMs
4. Generates milestone predictions for each player
5. Displays interpretable parameters for all GMs
"""
import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from chess_pinn_mvp.utils.data_manager import get_player_rating_data
from chess_pinn_mvp.utils.data_processor import prepare_training_data_for_player, RatingDataset
from chess_pinn_mvp.models.linear_fp_pinn import LinearFokkerPlanckPINN
from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from chess_pinn_mvp.models.trainer import PINNTrainer
from chess_pinn_mvp.visualization.visualizer import (
    plot_rating_history, 
    plot_comparative_trajectories,
    plot_rating_gains,
    plot_milestone_achievement_rates,
    plot_parameter_comparison,
    plot_model_comparison,
    create_summary_dashboard
)
from chess_pinn_mvp.utils.gm_config import GRANDMASTERS, GM_DICT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_argparse() -> argparse.ArgumentParser:
    """Setup argument parser for the application."""
    parser = argparse.ArgumentParser(description='Chess Rating Forecasting with PINNs')
    
    parser.add_argument(
        '--gms', 
        nargs='+', 
        default=['GM001', 'GM002', 'GM003', 'GM004', 'GM005', 'GM006', 'GM007'],
        help='List of GM keys to include (e.g., GM001 GM003)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=20, 
        help='Number of epochs to train each model'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=16, 
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--only-linear', 
        action='store_true',
        help='Train only linear models (faster)'
    )
    
    parser.add_argument(
        '--only-nonlinear', 
        action='store_true',
        help='Train only nonlinear models (more accurate but slower)'
    )
    
    parser.add_argument(
        '--no-train', 
        action='store_true',
        help='Skip training and use saved models if available'
    )
    
    parser.add_argument(
        '--forecast-years', 
        type=float, 
        default=5.0, 
        help='Number of years to forecast'
    )
    
    parser.add_argument(
        '--milestone-start', 
        type=int, 
        default=2300, 
        help='Starting milestone rating'
    )
    
    parser.add_argument(
        '--milestone-end', 
        type=int, 
        default=2900, 
        help='Ending milestone rating'
    )
    
    parser.add_argument(
        '--milestone-step', 
        type=int, 
        default=100, 
        help='Step size between milestones'
    )
    
    return parser


def train_models_for_gm(
    gm_key: str,
    epochs: int = 20,
    batch_size: int = 16,
    train_linear: bool = True,
    train_nonlinear: bool = True,
    no_train: bool = False
) -> Dict[str, Any]:
    """
    Train linear and/or nonlinear models for a grandmaster.
    
    Args:
        gm_key: GM key (e.g., 'GM001')
        epochs: Number of training epochs
        batch_size: Batch size for training
        train_linear: Whether to train linear model
        train_nonlinear: Whether to train nonlinear model
        no_train: Skip training and use saved models if available
        
    Returns:
        Dictionary with trained models and data
    """
    # Get GM info
    if gm_key not in GM_DICT:
        logger.error(f"Unknown GM key: {gm_key}")
        return {}
    
    gm_info = GM_DICT[gm_key]
    fide_id = gm_info['id']
    name = gm_info['name']
    
    logger.info(f"Processing data for {name} (FIDE ID: {fide_id})")
    
    # Load rating data
    df = get_player_rating_data(fide_id)
    
    if df.empty:
        logger.error(f"No data found for {name}")
        return {}
    
    logger.info(f"Loaded {len(df)} data points from {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
    
    # Plot rating history
    hist_path = os.path.join(FIGURES_DIR, f"{gm_key}_history.png")
    plot_rating_history(
        df=df,
        title=f"{name}: Rating History",
        save_path=hist_path
    )
    
    # Prepare training data
    train_data, test_data = prepare_training_data_for_player(
        fide_id=fide_id,
        test_fraction=0.2
    )
    
    if train_data is None:
        logger.error(f"Failed to prepare training data for {name}")
        return {}
    
    # Create datasets
    train_dataset = RatingDataset(train_data)
    val_dataset = RatingDataset(test_data) if test_data else None
    
    # Get initial rating
    r0 = df['rating'].iloc[0]
    
    # Initialize result dictionary
    result = {
        'gm_key': gm_key,
        'name': name,
        'df': df,
        'r0': r0,
        'linear_model': None,
        'nonlinear_model': None,
        'linear_trainer': None,
        'nonlinear_trainer': None
    }
    
    # Define model paths
    linear_model_path = os.path.join(MODELS_DIR, f"{gm_key}_linear_best.pt")
    nonlinear_model_path = os.path.join(MODELS_DIR, f"{gm_key}_nonlinear_best.pt")
    
    # Train/load linear model
    if train_linear:
        logger.info(f"Processing linear model for {name}")
        
        # Initialize model
        linear_model = LinearFokkerPlanckPINN(
            initial_alpha=0.005,  # Start with low mean-reversion
            initial_sigma=30.0,   # Reasonable volatility
            initial_mu_eq=df['rating'].max() + 100  # Slightly above max observed
        )
        
        # Initialize trainer
        linear_trainer = PINNTrainer(
            model=linear_model,
            lr=1e-3,
            physics_weight=0.5,
            device=DEVICE,
            output_dir=OUTPUT_DIR,
            model_name=f"{gm_key}_linear"
        )
        
        # Try to load saved model or train new one
        if no_train and os.path.exists(linear_model_path):
            logger.info(f"Loading saved linear model for {name}")
            linear_trainer.load_model(linear_model_path)
        else:
            logger.info(f"Training linear model for {name}")
            linear_trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=epochs // 2,
                verbose=True
            )
        
        # Plot results
        linear_trainer.plot_training_history(
            save_path=os.path.join(FIGURES_DIR, f"{gm_key}_linear_training.png")
        )
        linear_trainer.plot_rating_forecast(
            r0=r0,
            save_path=os.path.join(FIGURES_DIR, f"{gm_key}_linear_forecast.png")
        )
        
        # Add to result
        result['linear_model'] = linear_model
        result['linear_trainer'] = linear_trainer
    
    # Train/load nonlinear model
    if train_nonlinear:
        logger.info(f"Processing nonlinear model for {name}")
        
        # Initialize model
        nonlinear_model = NonlinearFokkerPlanckPINN(
            initial_alpha=0.005,      # Start with low mean-reversion
            initial_gamma=2.0,        # Start with quadratic nonlinearity
            initial_sigma0=30.0,      # Reasonable base volatility
            initial_beta=0.001,       # Low volatility decay
            initial_mu_eq=df['rating'].max() + 200,  # Higher than linear model
            initial_r0=r0             # Reference rating for volatility
        )
        
        # Initialize trainer
        nonlinear_trainer = PINNTrainer(
            model=nonlinear_model,
            lr=5e-4,  # Lower learning rate for stability
            physics_weight=0.6,  # Higher physics weight for nonlinear model
            device=DEVICE,
            output_dir=OUTPUT_DIR,
            model_name=f"{gm_key}_nonlinear"
        )
        
        # Try to load saved model or train new one
        if no_train and os.path.exists(nonlinear_model_path):
            logger.info(f"Loading saved nonlinear model for {name}")
            nonlinear_trainer.load_model(nonlinear_model_path)
        else:
            logger.info(f"Training nonlinear model for {name}")
            nonlinear_trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=epochs // 2,
                verbose=True
            )
        
        # Plot results
        nonlinear_trainer.plot_training_history(
            save_path=os.path.join(FIGURES_DIR, f"{gm_key}_nonlinear_training.png")
        )
        nonlinear_trainer.plot_rating_forecast(
            r0=r0,
            save_path=os.path.join(FIGURES_DIR, f"{gm_key}_nonlinear_forecast.png")
        )
        
        # Add to result
        result['nonlinear_model'] = nonlinear_model
        result['nonlinear_trainer'] = nonlinear_trainer
    
    # Compare models if both are available
    if train_linear and train_nonlinear:
        logger.info(f"Comparing linear and nonlinear models for {name}")
        plot_model_comparison(
            linear_model=linear_model,
            nonlinear_model=nonlinear_model,
            r0=r0,
            title=f"{name}: Linear vs. Nonlinear Model Comparison",
            save_path=os.path.join(FIGURES_DIR, f"{gm_key}_model_comparison.png")
        )
    
    return result


def create_comparative_analysis(
    gm_results: Dict[str, Dict[str, Any]],
    forecast_years: float = 5.0,
    milestone_start: int = 2300,
    milestone_end: int = 2900,
    milestone_step: int = 100,
    model_type: str = 'nonlinear'  # 'linear' or 'nonlinear'
) -> None:
    """
    Create comparative analysis visualizations.
    
    Args:
        gm_results: Dictionary mapping GM keys to their results
        forecast_years: Number of years to forecast
        milestone_start: Starting milestone rating
        milestone_end: Ending milestone rating
        milestone_step: Step size between milestones
        model_type: Model type to use ('linear' or 'nonlinear')
    """
    if not gm_results:
        logger.error("No GM results to compare")
        return
    
    # Get model key based on type
    model_key = f"{model_type}_model"
    
    # Prepare data for comparison
    gm_data = []
    for gm_key, result in gm_results.items():
        if model_key not in result or result[model_key] is None:
            logger.warning(f"No {model_type} model for {result['name']}")
            continue
        
        # Add to visualization data
        gm_data.append({
            'model': result[model_key],
            'name': result['name'],
            'r0': result['r0'],
            'color': None  # Let the visualization choose colors
        })
    
    if not gm_data:
        logger.error(f"No {model_type} models available for comparison")
        return
    
    # 1. Comparative trajectories
    logger.info(f"Creating comparative trajectory plot ({model_type} models)")
    plot_comparative_trajectories(
        gm_data=gm_data,
        title=f"Comparative Rating Trajectories ({model_type.capitalize()} Models)",
        t_max=365 * forecast_years,
        save_path=os.path.join(FIGURES_DIR, f"comparative_trajectories_{model_type}.png")
    )
    
    # 2. Rating gains
    logger.info(f"Creating rating gains plot ({model_type} models)")
    plot_rating_gains(
        gm_data=gm_data,
        title=f"Projected Rating Gains ({model_type.capitalize()} Models)",
        t_max=365 * forecast_years,
        save_path=os.path.join(FIGURES_DIR, f"rating_gains_{model_type}.png")
    )
    
    # 3. Milestone achievement rates
    logger.info(f"Creating milestone achievement plot ({model_type} models)")
    milestones = list(range(milestone_start, milestone_end + 1, milestone_step))
    plot_milestone_achievement_rates(
        gm_data=gm_data,
        title=f"Rating Milestone Achievement Rates ({model_type.capitalize()} Models)",
        milestones=milestones,
        save_path=os.path.join(FIGURES_DIR, f"milestone_rates_{model_type}.png")
    )
    
    # 4. Parameter comparison
    logger.info(f"Creating parameter comparison plot ({model_type} models)")
    plot_parameter_comparison(
        gm_data=gm_data,
        title=f"Model Parameter Comparison ({model_type.capitalize()} Models)",
        save_path=os.path.join(FIGURES_DIR, f"parameter_comparison_{model_type}.png")
    )


def main() -> None:
    """Main function."""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set up training flags
    train_linear = not args.only_nonlinear
    train_nonlinear = not args.only_linear
    
    # Log start
    logger.info(f"Starting Chess-PINNs-Forecasting on {DEVICE}")
    logger.info(f"Training models for {len(args.gms)} grandmasters")
    logger.info(f"Linear models: {train_linear}, Nonlinear models: {train_nonlinear}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Process each GM
    gm_results = {}
    for gm_key in args.gms:
        if gm_key not in GM_DICT:
            logger.warning(f"Unknown GM key: {gm_key}, skipping")
            continue
        
        logger.info(f"Processing {GM_DICT[gm_key]['name']} ({gm_key})")
        result = train_models_for_gm(
            gm_key=gm_key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_linear=train_linear,
            train_nonlinear=train_nonlinear,
            no_train=args.no_train
        )
        
        if result:
            gm_results[gm_key] = result
    
    # Create comparative analysis
    if train_linear:
        create_comparative_analysis(
            gm_results=gm_results,
            forecast_years=args.forecast_years,
            milestone_start=args.milestone_start,
            milestone_end=args.milestone_end,
            milestone_step=args.milestone_step,
            model_type='linear'
        )
    
    if train_nonlinear:
        create_comparative_analysis(
            gm_results=gm_results,
            forecast_years=args.forecast_years,
            milestone_start=args.milestone_start,
            milestone_end=args.milestone_end,
            milestone_step=args.milestone_step,
            model_type='nonlinear'
        )
    
    # Print summary
    logger.info("\n===== Summary of Results =====")
    for gm_key, result in gm_results.items():
        name = result['name']
        logger.info(f"\n{name} ({gm_key}):")
        
        if 'linear_model' in result and result['linear_model'] is not None:
            linear_params = result['linear_model'].get_parameters()
            logger.info("  Linear model parameters:")
            logger.info(f"    α (mean-reversion): {linear_params['alpha']:.6f}")
            logger.info(f"    σ (volatility): {linear_params['sigma']:.2f}")
            logger.info(f"    μ_eq (equilibrium): {linear_params['mu_eq']:.2f}")
        
        if 'nonlinear_model' in result and result['nonlinear_model'] is not None:
            nonlinear_params = result['nonlinear_model'].get_parameters()
            logger.info("  Nonlinear model parameters:")
            logger.info(f"    α (mean-reversion): {nonlinear_params['alpha']:.6f}")
            logger.info(f"    γ (nonlinearity): {nonlinear_params['gamma']:.2f}")
            logger.info(f"    σ₀ (base volatility): {nonlinear_params['sigma0']:.2f}")
            logger.info(f"    β (volatility decay): {nonlinear_params['beta']:.6f}")
            logger.info(f"    μ_eq (equilibrium): {nonlinear_params['mu_eq']:.2f}")
    
    logger.info("\nAnalysis completed successfully")
    logger.info(f"All results are stored in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
