"""
Very simple milestone evolution plot without PINN models.

This script creates a basic visualization of how milestone predictions
might evolve over a player's career, using only matplotlib and pandas.
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
output_path = output_dir / "simple_milestone_evolution.png"

print(f"Output directory: {output_dir}")
print(f"Output will be saved to: {output_path}")

# Create synthetic data
start_date = datetime(2015, 1, 1)
dates = [start_date + timedelta(days=30*i) for i in range(60)]  # 5 years of monthly data
ratings = [2400 + min(i*3, 300) - max(0, (i-50)*5) for i in range(60)]  # Growth then plateau

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'rating': ratings
})

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['rating'], 'ko-', label='Rating History')

# Add milestone lines
for milestone in [2500, 2600, 2700]:
    plt.axhline(milestone, color='gray', linestyle='--', alpha=0.5)
    plt.text(df['date'].iloc[-1], milestone, f' {milestone}', va='center')

# Format plot
plt.title('Simple Milestone Evolution Plot')
plt.xlabel('Date')
plt.ylabel('Rating')
plt.grid(True, alpha=0.3)
plt.legend()

# Format dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# Tight layout and save
plt.tight_layout()
print("Saving figure...")
plt.savefig(output_path, dpi=100)
plt.close()

print(f"Figure saved to {output_path}")
print(f"Checking if file exists: {os.path.exists(output_path)}")
