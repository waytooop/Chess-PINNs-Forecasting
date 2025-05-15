"""
Advanced visualization module for career milestone predictions with evolving forecasts.

This module provides specialized visualization tools for analyzing how forecasts
and milestone predictions evolve as a player's career progresses.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path

from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from chess_pinn_mvp.models.linear_fp_pinn import LinearFokkerPlanckPINN
from chess_pinn_mvp.utils.data_processor import prepare_training_data, RatingDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a color palette for prediction points
PREDICTION_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
]

def visualize_career_milestone_evolution(
    rating_history: List[Dict[str, Any]],
    prediction_points: List[int],
    milestones: List[int],
    output_path: str,
    model_type: str = 'nonlinear',
    forecast_years: int = 20,
    confidence_bands: bool = True,
    title: Optional[str] = None,
    epochs: int = 100
) -> None:
    """
    Create a comprehensive visualization of how milestone predictions evolve
    throughout a player's career.
    
    Args:
        rating_history: List of rating history points (date and rating)
        prediction_points: Points in the player's career (by index) to make predictions from
        milestones: Rating milestones to predict (e.g., [2200, 2400, 2600, 2800])
        output_path: Path to save the visualization
        model_type: Type of model to use ('linear' or 'nonlinear')
        forecast_years: Number of years to forecast from each prediction point
        confidence_bands: Whether to show confidence bands
        title: Custom title for the visualization
        epochs: Number of training epochs
    """
    # Convert rating history to DataFrame for easier handling
    dates = [datetime.strptime(entry['date'], '%Y-%m-%d') for entry in rating_history]
    ratings = [entry['rating'] for entry in rating_history]
    
    df = pd.DataFrame({
        'date': dates,
        'rating': ratings
    })
    df.set_index('date', inplace=True)
    df['t'] = (df.index - df.index.min()).days
    
    # Create figure (2 rows)
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Main plot for rating trajectory and forecasts
    ax_main = fig.add_subplot(gs[0])
    
    # Plot actual rating history
    ax_main.plot(df.index, df['rating'], 'k-', linewidth=2.5, label='Actual Rating History')
    ax_main.scatter(df.index, df['rating'], color='black', s=30, alpha=0.5)
    
    # Configure axes
    ax_main.set_ylabel('Rating')
    ax_main.set_title(title or 'Rating Trajectory with Evolving Milestone Predictions')
    ax_main.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Create secondary plot for milestone prediction changes
    ax_milestone = fig.add_subplot(gs[1], sharex=ax_main)
    ax_milestone.set_ylabel('Years to Achieve')
    ax_milestone.set_xlabel('Prediction Date')
    ax_milestone.grid(True, alpha=0.3)
    
    # Keep track of created milestone lines for the legend
    milestone_lines = {}
    
    # Dictionary to store milestone prediction data for the secondary plot
    milestone_predictions = {milestone: [] for milestone in milestones}
    
    # For each prediction point, train a model and make forecasts
    for i, prediction_idx in enumerate(prediction_points):
        if prediction_idx >= len(df):
            logger.warning(f"Prediction point {prediction_idx} exceeds data length, skipping")
            continue
        
        # Get date and rating at this prediction point
        prediction_date = df.index[prediction_idx]
        prediction_rating = df['rating'].iloc[prediction_idx]
        
        # Use data up to this point for training
        train_df = df.iloc[:prediction_idx+1].copy()
        
        # Log progress
        logger.info(f"Training model at point {i+1}/{len(prediction_points)}: "
                  f"Date: {prediction_date.date()}, Rating: {prediction_rating}")
        
        # Prepare training data
        train_data, _ = prepare_training_data(train_df, test_fraction=0)
        
        if train_data is None:
            logger.warning(f"Failed to prepare training data for point {i+1}, skipping")
            continue
        
        # Choose color for this prediction point
        color = PREDICTION_COLORS[i % len(PREDICTION_COLORS)]
        
        # Train model
        if model_type == 'nonlinear':
            model = NonlinearFokkerPlanckPINN(
                initial_alpha=0.005,
                initial_gamma=1.5,
                initial_sigma0=20.0,
                initial_beta=0.0005,
                initial_mu_eq=prediction_rating + 300,
                initial_r0=prediction_rating
            )
        else:
            model = LinearFokkerPlanckPINN(
                initial_alpha=0.005,
                initial_sigma=20.0,
                initial_mu_eq=prediction_rating + 200
            )
        
        # Create simple dataset
        dataset = RatingDataset(train_data)
        
        # Simple training loop (without trainer class for simplicity)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            # Get all data
            all_data = dataset.get_all_valid_data()
            t = all_data['t']
            r = all_data['r']
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            p_pred = model(r, t)
            
            # Simple data loss (negative log-likelihood)
            data_loss = -torch.log(p_pred + 1e-8).mean()
            
            # Generate some collocation points for PDE residual
            t_physics = torch.rand(1000) * torch.max(t).item()
            t_physics.requires_grad_(True)
            
            r_physics = torch.normal(mean=r.mean(), std=r.std(), size=(1000,))
            r_physics.requires_grad_(True)
            
            # Physics loss (PDE residual)
            pde_residual = model.fokker_planck_residual(r_physics, t_physics)
            physics_loss = torch.mean(pde_residual**2)
            
            # Combined loss
            total_loss = 0.4 * data_loss + 0.6 * physics_loss
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}, Loss: {total_loss.item():.6f}")
        
        # Generate forecast from this prediction point
        max_days = 365 * forecast_years
        t_values = np.linspace(0, max_days, 100)
        
        # Create date range for forecast
        forecast_dates = [prediction_date + timedelta(days=int(t)) for t in t_values]
        
        # Get forecast
        forecast = model.predict(r0=prediction_rating, t_values=t_values)
        
        # Plot mean forecast
        forecast_line = ax_main.plot(
            forecast_dates, forecast['mean'], '-', 
            color=color, 
            label=f'Forecast from {prediction_date.strftime("%Y-%m")} ({prediction_rating})'
        )
        
        # Plot confidence bands if requested
        if confidence_bands:
            ax_main.fill_between(
                forecast_dates, forecast['q25'], forecast['q75'],
                alpha=0.2, color=color
            )
        
        # Get milestone predictions and plot markers on the forecast line
        for milestone in milestones:
            # Only predict milestones that make sense given the current rating
            if (prediction_rating < milestone < model.mu_eq.item()) or \
               (prediction_rating > milestone > model.mu_eq.item()):
                
                # Get time to reach milestone
                days_to_milestone = forecast['milestones'].get(milestone, float('inf'))
                
                if not np.isinf(days_to_milestone) and not np.isnan(days_to_milestone):
                    # Convert to years
                    years_to_milestone = days_to_milestone / 365.0
                    
                    # Calculate milestone date
                    milestone_date = prediction_date + timedelta(days=int(days_to_milestone))
                    
                    # Only plot if within forecast range
                    if milestone_date <= forecast_dates[-1]:
                        # Plot milestone marker
                        marker = ax_main.scatter(
                            [milestone_date], [milestone],
                            marker='*', s=120, color=color, zorder=5,
                            edgecolor='black', linewidth=1
                        )
                        
                        # Add the milestone line to our tracking dict if not already there
                        if milestone not in milestone_lines:
                            milestone_lines[milestone] = ax_main.axhline(
                                milestone, linestyle='--', alpha=0.3, color='gray',
                                label=f'Milestone: {milestone}'
                            )
                        
                        # Add to milestone predictions for the secondary plot
                        milestone_predictions[milestone].append((
                            prediction_date, years_to_milestone, color
                        ))
    
    # Process and plot milestone predictions in the secondary plot
    for milestone, predictions in milestone_predictions.items():
        if predictions:
            # Extract data for plotting
            dates = [p[0] for p in predictions]
            years = [p[1] for p in predictions]
            colors = [p[2] for p in predictions]
            
            # Plot milestone prediction evolution
            ax_milestone.plot(
                dates, years, 'o-', 
                label=f'Time to {milestone}',
                color=colors[0]  # Use color of first prediction point
            )
            
            # Add milestone text
            ax_milestone.text(
                dates[-1], years[-1], f" {milestone}",
                verticalalignment='center', color=colors[0]
            )
    
    # Add legends
    ax_main.legend(loc='upper left', fontsize=9)
    # ax_milestone.legend(loc='upper right', fontsize=9)
    
    # Set y-axis limits for milestone plot based on data
    all_years = [y for preds in milestone_predictions.values() for _, y, _ in preds]
    if all_years:
        max_years = max(all_years) * 1.1
        ax_milestone.set_ylim(0, max_years)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Career milestone evolution visualization saved to {output_path}")
    plt.close(fig)


# Make this module runnable as a script
if __name__ == "__main__":
    import torch
    import json
    from pathlib import Path
    
    # Load example data (Magnus Carlsen)
    carlsen_id = "1503014"
    
    # Get the project root directory
    project_dir = Path(__file__).parent.parent
    
    # Define file paths
    data_file = project_dir / "data" / "raw" / f"{carlsen_id}_ratings.json"
    output_dir = project_dir / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(data_file, 'r') as f:
        carlsen_data = json.load(f)
    
    # Extract rating history
    rating_history = carlsen_data['rating_history']
    
    # Define prediction points (indices in the rating history)
    prediction_points = [0, 20, 40, 60, 80, 100]  # Roughly evenly spaced
    
    # Define milestones to predict
    milestones = [2500, 2600, 2700, 2800, 2900]
    
    # Create the visualization
    visualize_career_milestone_evolution(
        rating_history=rating_history,
        prediction_points=prediction_points,
        milestones=milestones,
        output_path=output_dir / "carlsen_milestone_evolution.png",
        model_type='nonlinear',
        forecast_years=10,
        confidence_bands=True,
        title="Magnus Carlsen: Rating Milestone Evolution",
        epochs=100
    )
    
    print(f"Visualization saved to {output_dir / 'carlsen_milestone_evolution.png'}")
