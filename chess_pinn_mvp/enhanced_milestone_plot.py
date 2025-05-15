"""
Enhanced milestone evolution visualization without full PINN models.

This script creates a comprehensive visualization of milestone predictions
that evolve throughout a player's career, using simplified models.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

# Ensure output directory exists
output_dir = Path(__file__).parent / "output" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "enhanced_milestone_evolution.png"

print(f"Output directory: {output_dir}")
print(f"Output will be saved to: {output_path}")

# Create synthetic data with more realistic rating progression
# (rapid growth, plateau, slight decline, etc.)
start_date = datetime(2015, 1, 1)
dates = [start_date + timedelta(days=30*i) for i in range(120)]  # 10 years of monthly data

# Create a more realistic career trajectory:
# - Initial rating of 2400
# - Rapid growth for first 3 years
# - Slower growth for next 2 years
# - Plateau around max rating
# - Slight decline near the end
def rating_formula(i):
    # Initial rating
    base = 2400
    
    # Rapid growth phase (first 36 months)
    if i < 36:
        return base + i * 8
    
    # Slower growth phase (next 24 months)
    elif i < 60:
        return base + 36 * 8 + (i - 36) * 3
    
    # Plateau phase (next 36 months)
    elif i < 96:
        return base + 36 * 8 + 24 * 3 + (i - 60) * 0.5
    
    # Slight decline phase
    else:
        return base + 36 * 8 + 24 * 3 + 36 * 0.5 - (i - 96) * 2

ratings = [rating_formula(i) for i in range(120)]

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'rating': ratings
})

# Create figure with two rows
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

# Main plot (top row)
ax_main = fig.add_subplot(gs[0])
ax_main.plot(df['date'], df['rating'], 'k-', linewidth=2.5, label='Actual Rating History')
ax_main.scatter(df['date'], df['rating'], color='black', s=30, alpha=0.5)

# Configure main axes
ax_main.set_ylabel('Rating')
ax_main.set_title('Chess Rating Milestone Evolution')
ax_main.grid(True, alpha=0.3)

# Format dates on x-axis
ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Milestone prediction plot (bottom row)
ax_milestone = fig.add_subplot(gs[1], sharex=ax_main)
ax_milestone.set_ylabel('Years to Achievement')
ax_milestone.set_xlabel('Prediction Date')
ax_milestone.grid(True, alpha=0.3)

# Define key milestones
milestones = [2500, 2600, 2700, 2800]

# Define prediction points (every 12 months for first 7 years)
prediction_points = [0, 12, 24, 36, 48, 60, 72, 84]
prediction_dates = [df['date'].iloc[idx] for idx in prediction_points]
prediction_ratings = [df['rating'].iloc[idx] for idx in prediction_points]

# Add milestone lines
milestone_lines = {}
for milestone in milestones:
    line = ax_main.axhline(milestone, color='gray', linestyle='--', alpha=0.3,
                          label=f'Milestone: {milestone}')
    milestone_lines[milestone] = line

# Define color palette for prediction points
colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
]

# Container for milestone predictions
milestone_predictions = {milestone: [] for milestone in milestones}

print("Processing prediction points...")

# For each prediction point, create a forecast and milestone predictions
for i, idx in enumerate(prediction_points):
    prediction_date = df['date'].iloc[idx]
    prediction_rating = df['rating'].iloc[idx]
    color = colors[i % len(colors)]
    
    print(f"  Point {i+1}: {prediction_date.date()}, Rating: {prediction_rating:.1f}")
    
    # Define a simple log growth model for forecasting
    # This simulates a player approaching an equilibrium rating
    def forecast_rating(t_days, current_rating, eq_rating=2800, alpha=0.001):
        # Simple exponential approach to equilibrium
        days = np.array(t_days)
        delta = eq_rating - current_rating
        return current_rating + delta * (1 - np.exp(-alpha * days))
    
    # Create forecast time points (5 years into future)
    forecast_days = np.arange(0, 365 * 5, 30)  # Every month for 5 years
    
    # Estimate equilibrium rating based on current rating
    # (The higher the current rating, the higher our equilibrium estimate)
    equilibrium = min(2850, prediction_rating + 400)
    
    # Generate forecast
    forecast_ratings = forecast_rating(
        forecast_days, 
        prediction_rating,
        eq_rating=equilibrium
    )
    
    # Add noise/uncertainty to create confidence bands
    # (More uncertainty for longer forecasts)
    uncertainty = np.array([5 + 0.1 * d for d in forecast_days])
    forecast_lower = forecast_ratings - uncertainty
    forecast_upper = forecast_ratings + uncertainty
    
    # Create forecast dates
    forecast_dates = [prediction_date + timedelta(days=int(d)) for d in forecast_days]
    
    # Plot forecast line
    forecast_line = ax_main.plot(
        forecast_dates, forecast_ratings, '-', 
        color=color, 
        label=f'Forecast from {prediction_date.strftime("%Y-%m")} ({prediction_rating:.0f})'
    )
    
    # Plot confidence bands
    ax_main.fill_between(
        forecast_dates, forecast_lower, forecast_upper,
        alpha=0.2, color=color
    )
    
    # For each milestone, calculate when it will be reached
    for milestone in milestones:
        # Skip milestones already achieved
        if prediction_rating >= milestone:
            continue
            
        # Skip milestones that won't be reached within forecast period
        if milestone > max(forecast_ratings):
            continue
        
        # Find the first point where forecast crosses milestone
        for j, r in enumerate(forecast_ratings):
            if r >= milestone:
                # Interpolate to find exact crossing point
                if j > 0:
                    t1, t2 = forecast_days[j-1], forecast_days[j]
                    r1, r2 = forecast_ratings[j-1], forecast_ratings[j]
                    # Linear interpolation
                    t_milestone = t1 + (t2 - t1) * (milestone - r1) / (r2 - r1)
                else:
                    t_milestone = forecast_days[j]
                
                # Convert to years
                years_to_milestone = t_milestone / 365.0
                
                # Calculate milestone date
                milestone_date = prediction_date + timedelta(days=int(t_milestone))
                
                # Plot milestone marker
                ax_main.scatter(
                    [milestone_date], [milestone],
                    marker='*', s=120, color=color, zorder=5,
                    edgecolor='black', linewidth=1
                )
                
                # Store milestone prediction for bottom plot
                milestone_predictions[milestone].append(
                    (prediction_date, years_to_milestone, color)
                )
                
                # We found the crossing point, move to next milestone
                break

# Plot milestone predictions in the secondary plot
print("Processing milestone predictions...")
for milestone, predictions in milestone_predictions.items():
    if predictions:
        # Extract data
        dates = [p[0] for p in predictions]
        years = [p[1] for p in predictions]
        colors = [p[2] for p in predictions]
        
        # Plot the evolution of time-to-milestone predictions
        ax_milestone.plot(
            dates, years, 'o-', 
            label=f'Time to {milestone}',
            color=colors[0]  # Use color of first prediction
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

# Save figure
print("Saving figure...")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Enhanced milestone evolution visualization saved to {output_path}")
