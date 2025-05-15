# Chess-PINNs Real Tournament Data

This component of the Chess-PINNs-Forecasting MVP uses real FIDE tournament data for accurate and realistic rating predictions.

## Why Real Data Matters

Using real historical FIDE tournament data provides several advantages:

1. **Authentic trajectories**: Real data captures the full complexity of chess careers including peaks, plateaus, and declines
2. **Tournament effects**: Real rating changes occur at discrete tournament events, reflecting actual competitive performance
3. **External factors**: Real data inherently incorporates external factors like player breaks, health issues, or competitive environment changes

By using real historical FIDE rating data, the PINN models can learn the true dynamics of chess rating evolution, resulting in accurate forecasts and meaningful parameter estimation.

## Available Functionality

This extension provides:

1. **Data structure**: A standard JSON format for storing real FIDE rating histories
2. **Data loader**: Utilities for loading, managing, and processing real tournament data
3. **Training pipeline**: Functions for training PINN models using real data
4. **Comparative analysis**: Tools for comparing multiple players and model types

## Usage Guide

### Adding Real Tournament Data

Real tournament data should be added to the `real_tournament_history.json` file, which follows the schema defined in `real_tournament_history_schema.json`. You can use the provided utility functions:

```python
from chess_pinn_mvp.utils.real_data_loader import add_player_ratings

# Example data for Magnus Carlsen
carlsen_ratings = [
    {"date": "2010-01-01", "rating": 2810},
    {"date": "2010-07-01", "rating": 2826},
    # Add more historical ratings...
]

# Add data
add_player_ratings("1503014", carlsen_ratings)
```

### Training Models with Real Data

The `demo_real_data.py` script demonstrates how to train models using real data:

```bash
python -m chess_pinn_mvp.demo_real_data
```

You can also use the functions directly in your own scripts:

```python
from chess_pinn_mvp.utils.real_data_loader import load_real_player_data
from chess_pinn_mvp.models.trainer import PINNTrainer
from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN

# Load player data
df = load_real_player_data("1503014")  # Magnus Carlsen's FIDE ID

# Create and train model
# ... (see demo_real_data.py for complete example)
```

### Improved Nonlinear Model Parameters

With real data, the nonlinear model parameters are more interpretable:

- `alpha`: Measures the rate of mean reversion - how quickly a player tends to return to their "equilibrium" rating
- `gamma`: Controls the nonlinearity of the drift term - values closer to 1 indicate more linear dynamics, higher values indicate stronger nonlinear effects
- `mu_eq`: The equilibrium rating - the player's long-term potential rating level
- `sigma0`: The base volatility of rating changes
- `beta`: The decay rate for volatility as rating increases

These parameters can provide insights into a player's rating dynamics and long-term potential.

## FIDE Data Sources

Real FIDE rating data can be obtained from:

1. Official FIDE rating website: https://ratings.fide.com/
2. FIDE profile pages for individual players (e.g., https://ratings.fide.com/profile/1503014/chart for Magnus Carlsen)
3. FIDE rating lists published monthly

The data should be formatted as a list of date/rating pairs, with dates in ISO format (YYYY-MM-DD).

## Limitations

Some limitations to be aware of:

1. The real data loader assumes regular, well-formatted rating data
2. The current implementation focuses on standard ratings only
3. Parameter estimation is more sensitive to data quality with real data
4. More epochs (200+) are recommended for training with real data

## Future Improvements

Potential enhancements:

1. Automated scraping of rating data from FIDE website
2. Support for rapid and blitz ratings
3. Handling of missing or irregular rating data
4. Integration of tournament performance data (not just rating changes)
5. Accounting for rating inflation over time
