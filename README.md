# Chess-PINNs-Forecasting

A physics-informed neural network (PINN) approach to chess rating forecasting, using the Fokker-Planck equation to model rating dynamics.

## Overview

This project implements a PINN to forecast chess player rating progression over time. The system models rating evolution as an Ornstein-Uhlenbeck stochastic process, governed by the Fokker-Planck equation. The model can predict:

- Future rating distributions
- Probability of reaching rating milestones
- Expected time to achieve target ratings

The system supports both Lichess API data and FIDE tournament performance data to train player-specific models and generate personalized forecasts.

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- DeepXDE 1.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Chess-PINNs-Forecasting.git
cd Chess-PINNs-Forecasting
```

2. Install the package:
```bash
pip install -e .
```

This will install the `chess-pinn` command-line tool and all dependencies.

## Usage

The package provides a command-line interface for common operations:

### Fetch Rating History

```bash
chess-pinn fetch --username DrNykterstein --time-control blitz
```

This fetches and processes the rating history for a player, generating a plot of their historical ratings.

### Train a Model

```bash
chess-pinn train --username DrNykterstein --time-control blitz
```

Trains a Fokker-Planck PINN model on the player's rating history. Additional options include:
- `--epochs` - Number of training epochs
- `--batch-size` - Batch size for training
- `--physics-weight` - Weight of physics constraint in the loss function

### Generate a Forecast

```bash
chess-pinn forecast --username DrNykterstein --time-control blitz --days 180
```

Generates a comprehensive forecast report including:
- Rating history visualization
- Probabilistic forecast with confidence intervals
- Milestone probability analysis

### Milestone Analysis

```bash
chess-pinn milestone --username DrNykterstein --target 2200
```

Calculates the probability of reaching a target rating over time.

You can also specify a target date:
```bash
chess-pinn milestone --username DrNykterstein --target 2200 --by-date 2025-12-31
```

## The Physics Behind the Model

The Fokker-Planck equation describes the time evolution of a probability density function under drift and diffusion:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial r}[\mu(r)p] + \frac{1}{2}\frac{\partial^2}{\partial r^2}[\sigma^2(r)p]$$

Where:
- $p(r,t)$ is the probability density of rating $r$ at time $t$
- $\mu(r) = -\alpha(r-\mu_{eq})$ is the drift term (mean reversion)
- $\sigma(r)$ is the diffusion coefficient (volatility)

The model learns these parameters from historical rating data while enforcing the physics constraint.

## Project Structure

- `chess_pinn/utils/data_fetcher.py` - API interaction with Lichess
- `chess_pinn/utils/data_processor.py` - Data processing and feature extraction
- `chess_pinn/models/fp_pinn.py` - Fokker-Planck PINN implementation
- `chess_pinn/utils/visualizer.py` - Visualization utilities
- `chess_pinn/cli.py` - Command-line interface
- `examples/fide_milestone_forecast.py` - Example script for FIDE milestone analysis