#!/usr/bin/env python3
"""
Simple demonstration script for Chess-PINNs-Forecasting MVP.

This script:
1. Loads synthetic rating data for Magnus Carlsen
2. Trains both linear and nonlinear PINN models
3. Compares the models and makes milestone predictions
4. Generates visualization dashboards
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from chess_pinn_mvp.utils.data_manager import get_player_rating_data
from chess_pinn_mvp.utils.data_processor import prepare_training_data_for_player, RatingDataset
from chess_pinn_mvp.models.linear_fp_pinn import LinearFokkerPlanckPINN
from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from chess_pinn_mvp.models.trainer import PINNTrainer
from chess_pinn_mvp.visualization.visualizer import plot_model_comparison, plot_rating_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FIDE_ID_CARLSEN = "1503014"  # Magnus Carlsen
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)

# Training parameters
EPOCHS = 5  # Use fewer epochs just for quick testing
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    """Main function."""
    logger.info(f"Starting simple demo using device: {DEVICE}")
    
    # 1. Load data for Magnus Carlsen
    logger.info("Loading data for Magnus Carlsen")
    df = get_player_rating_data(FIDE_ID_CARLSEN)
    
    if df.empty:
        logger.error("Failed to load data")
        return
    
    logger.info(f"Loaded {len(df)} data points from {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
    
    # Plot rating history
    logger.info("Plotting rating history")
    fig = plot_rating_history(
        df=df,
        title="Magnus Carlsen: Rating History",
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_history.png')
    )
    plt.close(fig)
    
    # 2. Prepare training data
    logger.info("Preparing training data")
    train_data, test_data = prepare_training_data_for_player(
        fide_id=FIDE_ID_CARLSEN,
        test_fraction=0.2
    )
    
    if train_data is None:
        logger.error("Failed to prepare training data")
        return
    
    # Create datasets
    train_dataset = RatingDataset(train_data)
    val_dataset = RatingDataset(test_data) if test_data else None
    
    # Get initial rating
    r0 = df['rating'].iloc[0]
    logger.info(f"Initial rating: {r0}")
    
    # 3. Train linear model
    logger.info("Training linear PINN model")
    linear_model = LinearFokkerPlanckPINN(
        initial_alpha=0.005,  # Start with low mean-reversion
        initial_sigma=30.0,   # Reasonable volatility
        initial_mu_eq=df['rating'].max() + 100  # Slightly above max observed
    )
    
    linear_trainer = PINNTrainer(
        model=linear_model,
        lr=1e-3,
        physics_weight=0.5,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        model_name="carlsen_linear"
    )
    
    linear_history = linear_trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=30,
        verbose=True
    )
    
    # Plot linear model training history and forecast
    logger.info("Plotting linear model results")
    linear_trainer.plot_training_history(
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_linear_training.png')
    )
    linear_trainer.plot_rating_forecast(
        r0=r0,
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_linear_forecast.png')
    )
    linear_trainer.plot_milestone_predictions(
        r0=r0,
        milestones=[2500, 2600, 2700, 2800],
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_linear_milestones.png')
    )
    
    # 4. Train nonlinear model
    logger.info("Training nonlinear PINN model")
    nonlinear_model = NonlinearFokkerPlanckPINN(
        initial_alpha=0.005,      # Start with low mean-reversion
        initial_gamma=2.0,        # Start with quadratic nonlinearity
        initial_sigma0=30.0,      # Reasonable base volatility
        initial_beta=0.001,       # Low volatility decay
        initial_mu_eq=df['rating'].max() + 200,  # Higher than linear model
        initial_r0=r0             # Reference rating for volatility
    )
    
    nonlinear_trainer = PINNTrainer(
        model=nonlinear_model,
        lr=5e-4,  # Lower learning rate for stability
        physics_weight=0.6,  # Higher physics weight for nonlinear model
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        model_name="carlsen_nonlinear"
    )
    
    nonlinear_history = nonlinear_trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=30,
        verbose=True
    )
    
    # Plot nonlinear model training history and forecast
    logger.info("Plotting nonlinear model results")
    nonlinear_trainer.plot_training_history(
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_nonlinear_training.png')
    )
    nonlinear_trainer.plot_rating_forecast(
        r0=r0,
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_nonlinear_forecast.png')
    )
    nonlinear_trainer.plot_milestone_predictions(
        r0=r0,
        milestones=[2500, 2600, 2700, 2800],
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_nonlinear_milestones.png')
    )
    
    # 5. Compare the two models
    logger.info("Comparing linear and nonlinear models")
    fig = plot_model_comparison(
        linear_model=linear_model,
        nonlinear_model=nonlinear_model,
        r0=r0,
        title="Magnus Carlsen: Linear vs. Nonlinear Model Comparison",
        save_path=os.path.join(OUTPUT_DIR, 'figures', 'carlsen_model_comparison.png')
    )
    plt.close(fig)
    
    # Print final parameters
    logger.info("Final model parameters:")
    
    linear_params = linear_model.get_parameters()
    logger.info("\nLinear model parameters:")
    logger.info(f"  Alpha (mean-reversion): {linear_params['alpha']:.6f}")
    logger.info(f"  Sigma (volatility): {linear_params['sigma']:.2f}")
    logger.info(f"  Mu_eq (equilibrium): {linear_params['mu_eq']:.2f}")
    
    nonlinear_params = nonlinear_model.get_parameters()
    logger.info("\nNonlinear model parameters:")
    logger.info(f"  Alpha (mean-reversion): {nonlinear_params['alpha']:.6f}")
    logger.info(f"  Gamma (nonlinearity): {nonlinear_params['gamma']:.2f}")
    logger.info(f"  Sigma0 (base volatility): {nonlinear_params['sigma0']:.2f}")
    logger.info(f"  Beta (volatility decay): {nonlinear_params['beta']:.6f}")
    logger.info(f"  Mu_eq (equilibrium): {nonlinear_params['mu_eq']:.2f}")
    
    logger.info("\nDemo completed successfully")
    logger.info(f"All results are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
