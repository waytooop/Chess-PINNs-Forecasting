"""
Simple test script for the milestone evolution visualization.

This script provides a minimal example to test the milestone evolution
visualization functionality with synthetic data.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from chess_pinn_mvp.models.linear_fp_pinn import LinearFokkerPlanckPINN
from chess_pinn_mvp.utils.data_processor import RatingDataset

# Set up output directory
output_dir = Path(__file__).parent / "output" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "test_milestone_evolution.png"

# Create synthetic rating history
start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 1, 1)
num_points = 60  # 5 years of bi-monthly ratings

# Generate dates and ratings
dates = [start_date + timedelta(days=(end_date - start_date).days / (num_points - 1) * i) 
         for i in range(num_points)]
ratings = [2400 + i * 2 for i in range(num_points)]  # Simple linear progression

# Convert to dataframe
df = pd.DataFrame({
    'date': dates,
    'rating': ratings
})
df.set_index('date', inplace=True)
df['t'] = (df.index - df.index.min()).days

# Create figure (2 rows)
print("Creating visualization...")
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

# Main plot for rating trajectory and forecasts
ax_main = fig.add_subplot(gs[0])

# Plot actual rating history
ax_main.plot(df.index, df['rating'], 'k-', linewidth=2.5, label='Actual Rating History')
ax_main.scatter(df.index, df['rating'], color='black', s=30, alpha=0.5)

# Configure axes
ax_main.set_ylabel('Rating')
ax_main.set_title('Test Rating Trajectory')
ax_main.grid(True, alpha=0.3)

# Create secondary plot for milestone prediction changes
ax_milestone = fig.add_subplot(gs[1], sharex=ax_main)
ax_milestone.set_ylabel('Years to Achieve')
ax_milestone.set_xlabel('Prediction Date')
ax_milestone.grid(True, alpha=0.3)

# Define prediction points and milestones
prediction_points = [0, 15, 30, 45]  # 4 points evenly spaced
milestones = [2450, 2500, 2550]

# Define a color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Container for milestone predictions
milestone_predictions = {milestone: [] for milestone in milestones}

print("Processing prediction points...")
# For each prediction point
for i, idx in enumerate(prediction_points):
    # Get prediction date and rating
    prediction_date = df.index[idx]
    prediction_rating = df['rating'].iloc[idx]
    color = colors[i]
    
    print(f"  Point {i+1}: {prediction_date.date()}, Rating: {prediction_rating}")
    
    # Create and train a simple model 
    model = NonlinearFokkerPlanckPINN(
        initial_alpha=0.005,
        initial_gamma=1.5,
        initial_sigma0=20.0,
        initial_beta=0.0005,
        initial_mu_eq=2700,  # Fixed target for test
        initial_r0=prediction_rating
    )
    
    # Generate forecast
    forecast_years = 5
    max_days = 365 * forecast_years
    t_values = np.linspace(0, max_days, 100)
    
    # Create date range for forecast
    forecast_dates = [prediction_date + timedelta(days=int(t)) for t in t_values]
    
    # Generate a simple linear forecast for testing (no need for actual model prediction)
    forecast_mean = np.linspace(prediction_rating, 2600, len(t_values))
    forecast_q25 = forecast_mean - 20
    forecast_q75 = forecast_mean + 20
    
    # Plot forecast
    ax_main.plot(
        forecast_dates, forecast_mean, '-', 
        color=color, 
        label=f'Forecast from {prediction_date.strftime("%Y-%m")} ({prediction_rating})'
    )
    
    # Plot confidence band
    ax_main.fill_between(
        forecast_dates, forecast_q25, forecast_q75,
        alpha=0.2, color=color
    )
    
    # Generate milestone predictions manually
    for milestone in milestones:
        if prediction_rating < milestone:  # Only predict future milestones
            # Simple linear interpolation to find days to milestone
            days_to_milestone = ((milestone - prediction_rating) / 
                               (forecast_mean[-1] - prediction_rating) * max_days)
            
            if days_to_milestone <= max_days:
                # Convert to years
                years_to_milestone = days_to_milestone / 365.0
                
                # Calculate milestone date
                milestone_date = prediction_date + timedelta(days=int(days_to_milestone))
                
                # Plot milestone marker
                ax_main.scatter(
                    [milestone_date], [milestone],
                    marker='*', s=120, color=color, zorder=5,
                    edgecolor='black', linewidth=1
                )
                
                # Add horizontal line for milestone
                if not any(m.get_ydata()[0] == milestone for m in ax_main.get_lines() 
                          if isinstance(m, plt.Line2D) and len(m.get_ydata()) > 0):
                    ax_main.axhline(
                        milestone, linestyle='--', alpha=0.3, color='gray',
                        label=f'Milestone: {milestone}'
                    )
                
                # Store for bottom plot
                milestone_predictions[milestone].append((
                    prediction_date, years_to_milestone, color
                ))

print("Processing milestone predictions...")
# Plot milestone prediction evolution in bottom plot
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
            color=colors[0]
        )
        
        # Add milestone text
        ax_milestone.text(
            dates[-1], years[-1], f" {milestone}",
            verticalalignment='center', color=colors[0]
        )

# Add legends
ax_main.legend(loc='upper left', fontsize=9)

# Set y-axis limits for milestone plot
all_years = [y for preds in milestone_predictions.values() for _, y, _ in preds]
if all_years:
    max_years = max(all_years) * 1.1
    ax_milestone.set_ylim(0, max_years)

print("Saving visualization...")
# Save figure
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Test visualization saved to {output_path}")
