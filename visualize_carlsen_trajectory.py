"""
Visualize Magnus Carlsen's rating trajectory and milestone achievements
using the detailed rating data from real_tournament_history_schema.json
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

# Import necessary modules from the project
from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from chess_pinn_mvp.models.trainer import FokkerPlanckTrainer

# Set paths and constants
DATA_DIR = os.path.join('chess_pinn_mvp', 'data')
SCHEMA_FILE = os.path.join(DATA_DIR, 'real_tournament_history_schema.json')
OUTPUT_DIR = os.path.join('output', 'carlsen_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the schema data
def load_schema_data():
    with open(SCHEMA_FILE, 'r') as f:
        data = json.load(f)
    return data

# Process Carlsen's data into a DataFrame
def process_carlsen_data(data):
    # Find Carlsen in the players list
    carlsen_data = None
    for player in data['players']:
        if player['name'] == 'Magnus Carlsen':
            carlsen_data = player
            break
    
    if not carlsen_data:
        raise ValueError("Magnus Carlsen not found in the data")
    
    # Convert rating data to DataFrame
    ratings = []
    for rating_info in carlsen_data['standard_ratings']:
        if 'period' in rating_info:  # Schema file format
            date_str = rating_info['period']
            date = datetime.strptime(date_str, '%Y-%b')
            rating = rating_info['rating']
        else:  # Other format
            date_str = rating_info['date']
            date = datetime.strptime(date_str, '%Y-%m-%d')
            rating = rating_info['rating']
        
        ratings.append({'date': date, 'rating': rating})
    
    df = pd.DataFrame(ratings)
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    
    # Add time since start in days
    df['t'] = (df.index - df.index.min()).days.astype(float)
    
    return df

# Extract milestone dates - when Carlsen reached specific rating levels
def extract_milestones(df, milestone_ratings=[2200, 2300, 2400, 2500, 2600, 2700, 2800]):
    milestones = {}
    
    for milestone in milestone_ratings:
        # Find the first date when Carlsen reached or exceeded this rating
        reached = df[df['rating'] >= milestone]
        if not reached.empty:
            first_date = reached.index.min()
            milestones[milestone] = {
                'date': first_date,
                'actual_rating': df.loc[first_date, 'rating'],
                'days_since_start': df.loc[first_date, 't']
            }
    
    return milestones

# Plot Carlsen's rating trajectory with milestones
def plot_trajectory(df, milestones, output_path):
    plt.figure(figsize=(12, 8))
    
    # Plot actual ratings
    plt.plot(df.index, df['rating'], 'b-', label='Actual Rating', linewidth=2)
    
    # Mark the milestones
    milestone_dates = []
    milestone_ratings = []
    milestone_labels = []
    
    for rating, info in milestones.items():
        milestone_dates.append(info['date'])
        milestone_ratings.append(info['actual_rating'])
        milestone_labels.append(f"{rating}")
    
    plt.scatter(milestone_dates, milestone_ratings, color='red', s=100, zorder=5)
    
    # Add milestone labels
    for i, label in enumerate(milestone_labels):
        plt.annotate(f"{label}", 
                    (milestone_dates[i], milestone_ratings[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8))
    
    # Add formatted x-axis
    years = mdates.YearLocator()
    years_fmt = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_locator(years)
    plt.gca().xaxis.set_major_formatter(years_fmt)
    
    # Add reference horizontal lines for major rating thresholds
    for rating in range(2200, 2900, 100):
        plt.axhline(y=rating, color='grey', linestyle='--', alpha=0.3)
    
    plt.title('Magnus Carlsen Rating Trajectory', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('FIDE Rating', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Trajectory plot saved to {output_path}")

# Generate table of milestone achievements
def generate_milestone_table(milestones):
    # Convert milestone data to a DataFrame for easy display
    milestone_data = []
    for rating, info in milestones.items():
        milestone_data.append({
            'Milestone': rating,
            'Date Reached': info['date'].strftime('%Y-%m-%d'),
            'Actual Rating': info['actual_rating'],
            'Days Since First Rating': int(info['days_since_start'])
        })
    
    milestone_df = pd.DataFrame(milestone_data)
    milestone_df.sort_values('Milestone', inplace=True)
    
    return milestone_df

# Add forecasted trajectory starting from 2200
def predict_trajectory(r0=2200, t_days=3650):  # ~10 years forecast
    # Initialize the model with reasonable parameters for chess ratings
    model = NonlinearFokkerPlanckPINN(
        hidden_layers=[64, 128, 128, 64],
        initial_alpha=0.01,    # Mean reversion rate
        initial_gamma=1.5,     # Nonlinearity exponent 
        initial_sigma0=30.0,   # Base volatility
        initial_beta=0.001,    # Volatility decay rate
        initial_mu_eq=2850.0,  # Equilibrium rating (Carlsen's peak was ~2882)
        initial_r0=r0          # Reference rating for volatility
    )
    
    # Create time values in days
    t_values = np.linspace(0, t_days, 100)
    
    # Get the predictions
    predictions = model.predict(r0=r0, t_values=t_values, use_numerical=True)
    
    return predictions, t_values

# Plot both actual and forecasted trajectories
def plot_comparison(df, milestones, r0=2200, output_path=None):
    # Get prediction starting from r0
    predictions, t_values = predict_trajectory(r0=r0)
    
    # Find the start date for the prediction (when Carlsen first reached r0)
    reached_r0 = df[df['rating'] >= r0]
    if reached_r0.empty:
        print(f"Carlsen never reached rating {r0} in the data")
        return
    
    start_date = reached_r0.index.min()
    t_days_passed = df.loc[start_date, 't']
    
    # Create prediction dates
    from pandas import TimedeltaIndex
    earliest_date = df.index.min()
    pred_dates = [earliest_date + pd.Timedelta(days=float(t_days_passed + t)) for t in t_values]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual ratings
    plt.plot(df.index, df['rating'], 'b-', label='Actual Rating', linewidth=2)
    
    # Plot predictions
    plt.plot(pred_dates, predictions['mean'], 'r--', label='Predicted Mean', linewidth=2)
    plt.fill_between(pred_dates, predictions['q05'], predictions['q95'], 
                    color='red', alpha=0.2, label='90% Confidence Interval')
    
    # Mark the milestones
    milestone_dates = []
    milestone_ratings = []
    milestone_labels = []
    
    for rating, info in sorted(milestones.items()):
        if rating >= r0:  # Only show milestones from r0 onwards
            milestone_dates.append(info['date'])
            milestone_ratings.append(info['actual_rating'])
            milestone_labels.append(f"{rating}")
    
    plt.scatter(milestone_dates, milestone_ratings, color='green', s=100, zorder=5)
    
    # Add milestone labels
    for i, label in enumerate(milestone_labels):
        plt.annotate(f"{label}", 
                   (milestone_dates[i], milestone_ratings[i]),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8))
    
    # Add vertical line at prediction start date
    plt.axvline(x=start_date, color='purple', linestyle='-', alpha=0.7, 
               label=f'Forecast Start ({start_date.strftime("%Y-%m-%d")}, Rating: {r0})')
    
    # Add formatted x-axis
    years = mdates.YearLocator()
    years_fmt = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_locator(years)
    plt.gca().xaxis.set_major_formatter(years_fmt)
    
    # Add reference horizontal lines for major rating thresholds
    for rating in range(2200, 2900, 100):
        plt.axhline(y=rating, color='grey', linestyle='--', alpha=0.3)
    
    plt.title(f'Magnus Carlsen Rating Trajectory vs Forecast (Starting at {r0})', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('FIDE Rating', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Save the figure
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Comparison plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

# Compare predicted milestone timing with actual
def compare_milestones(df, r0=2200):
    # Get prediction starting from r0
    predictions, _ = predict_trajectory(r0=r0)
    
    # Get actual milestone dates
    milestone_ratings = sorted([m for m in range(2300, 2900, 100) if m > r0])
    actual_milestones = extract_milestones(df, milestone_ratings)
    
    # Get predicted milestone timings in days
    predicted_milestones = {}
    for milestone, days in predictions['milestones'].items():
        if int(milestone) in milestone_ratings and days < float('inf'):
            predicted_milestones[int(milestone)] = days
    
    # Find the start date when Carlsen reached r0
    reached_r0 = df[df['rating'] >= r0]
    if reached_r0.empty:
        print(f"Carlsen never reached rating {r0} in the data")
        return None
    
    start_date = reached_r0.index.min()
    t_days_passed = df.loc[start_date, 't']
    
    # Compare actual vs predicted
    comparison = []
    for rating in milestone_ratings:
        if rating in actual_milestones and rating in predicted_milestones:
            actual_days = actual_milestones[rating]['days_since_start'] - t_days_passed
            predicted_days = predicted_milestones[rating]
            
            comparison.append({
                'Milestone': rating,
                'Actual Days': int(actual_days),
                'Predicted Days': int(predicted_days),
                'Difference': int(predicted_days - actual_days)
            })
    
    comparison_df = pd.DataFrame(comparison)
    return comparison_df

# Main function
def main():
    print("Starting Magnus Carlsen trajectory analysis...")
    
    # Load the data
    print("Loading schema data...")
    data = load_schema_data()
    print(f"Found {len(data['players'])} players in the schema data")
    
    # Process Carlsen's data
    print("Processing Magnus Carlsen's rating data...")
    df = process_carlsen_data(data)
    print(f"Loaded {len(df)} rating entries from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
    
    # Extract milestones
    print("Extracting rating milestones...")
    milestones = extract_milestones(df)
    print(f"Found {len(milestones)} milestones")
    
    # Plot trajectory
    trajectory_plot_path = os.path.join(OUTPUT_DIR, 'carlsen_trajectory.png')
    plot_trajectory(df, milestones, trajectory_plot_path)
    
    # Generate and display milestone table
    milestone_table = generate_milestone_table(milestones)
    print("\nMagnus Carlsen's Rating Milestones:")
    print(milestone_table)
    
    # Save milestone table to CSV
    milestone_csv_path = os.path.join(OUTPUT_DIR, 'carlsen_milestones.csv')
    milestone_table.to_csv(milestone_csv_path, index=False)
    print(f"Milestone table saved to {milestone_csv_path}")
    
    # Plot comparison with predictions from 2200 onwards
    comparison_plot_path = os.path.join(OUTPUT_DIR, 'carlsen_forecast_comparison_2200.png')
    plot_comparison(df, milestones, r0=2200, output_path=comparison_plot_path)
    
    # Compare milestone timings
    milestone_comparison = compare_milestones(df, r0=2200)
    if milestone_comparison is not None:
        print("\nMilestone Achievement Comparison (Days from 2200):")
        print(milestone_comparison)
        
        # Save comparison to CSV
        comparison_csv_path = os.path.join(OUTPUT_DIR, 'carlsen_milestone_comparison.csv')
        milestone_comparison.to_csv(comparison_csv_path, index=False)
        print(f"Milestone comparison saved to {comparison_csv_path}")

if __name__ == "__main__":
    main()
