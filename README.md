# Chess-PINNs-Forecasting

Physics-Informed Neural Networks (PINNs) for chess rating forecasting using the Fokker-Planck equation.

## Project Overview

This project applies physics-informed neural networks to the domain of chess rating forecasting. By modeling rating evolution using the Fokker-Planck equation, we can infer physical parameters from sparse rating data and forecast future rating trajectories.

The core innovation is using PINNs to solve both the forward problem (predicting future ratings given physics parameters) and the inverse problem (inferring physics parameters from observed rating data).

## Key Features

- **Physics-Based Modeling**: Uses the Fokker-Planck equation to model rating dynamics as a diffusion process
- **Dual Model Architecture**:
  - **Linear Model**: Ornstein-Uhlenbeck stochastic process with constant volatility
  - **Nonlinear Model**: "Chess Asymptotic" process with nonlinear mean-reversion and rating-dependent volatility
- **Visualization Tools**: Comprehensive visualization of model parameters, training processes, and forecasts
- **Real Data Analysis**: Works with real grandmaster FIDE rating histories
- **Comparative Analysis**: Tools to compare different models and players

## Project Structure

```
Chess-PINNs-Forecasting/
├── chess_pinn_mvp/               # Main package
│   ├── models/                   # PINN model implementations
│   │   ├── linear_fp_pinn.py     # Linear Fokker-Planck PINN
│   │   ├── nonlinear_fp_pinn.py  # Nonlinear Fokker-Planck PINN
│   │   └── trainer.py            # Training framework
│   ├── utils/                    # Utility functions
│   │   ├── data_processor.py     # Data processing utilities
│   │   ├── gm_config.py          # Grandmaster configuration
│   │   └── fide_data_manager.py  # FIDE data handling
│   └── visualization/            # Visualization utilities
│       ├── visualizer.py         # Core visualization functions
│       └── career_milestone_visualizer.py  # Milestone analysis
├── output/                       # Output directory for results
│   ├── inverse_problem_visualization/  # Visualization results
│   ├── models/                   # Saved model parameters
│   └── carlsen_analysis/         # Example player analysis
├── visualize_inverse_problem.py  # Main script for inverse problem visualization
└── visualize_carlsen_trajectory.py  # Specific analysis for Magnus Carlsen
```

## Technical Approach

### The Physics Model

The Fokker-Planck equation describes the time evolution of a probability density function under drift and diffusion:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial r}[\mu(r)p] + \frac{1}{2}\frac{\partial^2}{\partial r^2}[\sigma^2(r)p]$$

Where:
- $p(r,t)$ is the probability density of rating $r$ at time $t$
- $\mu(r)$ is the drift term modeling mean reversion
- $\sigma(r)$ is the diffusion coefficient modeling rating volatility

### Model Variants

1. **Linear Model**:
   - Drift: $\mu(r) = -\alpha(r-\mu_{eq})$ (linear mean reversion)
   - Diffusion: $\sigma(r) = \sigma$ (constant volatility)
   - Parameters: $\alpha$ (mean reversion rate), $\sigma$ (volatility), $\mu_{eq}$ (equilibrium rating)

2. **Nonlinear Model**:
   - Drift: $\mu(r) = -\alpha(r-\mu_{eq})^\gamma$ (nonlinear mean reversion)
   - Diffusion: $\sigma^2(r) = \sigma_0^2 e^{-\beta(r-r_0)}$ (rating-dependent volatility)
   - Parameters: $\alpha$, $\gamma$ (nonlinearity), $\sigma_0$ (base volatility), $\beta$ (volatility decay), $\mu_{eq}$, $r_0$ (reference rating)

## Running the Code

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- DeepXDE (for PINN implementations)

### Visualization Examples

To visualize the inverse problem solution:

```bash
python visualize_inverse_problem.py
```

This will:
1. Load grandmaster rating data
2. Train both linear and nonlinear models
3. Generate visualizations of the training process, model parameters, and forecasts
4. Save results to the output directory

### Model Training Process

The training process combines:
- **Data Loss**: Measuring how well the model fits observed rating data
- **Physics Loss**: Ensuring the solution satisfies the Fokker-Planck equation
- **Adaptive Weighting**: Balancing data and physics constraints

## Key Visualizations

The project generates several types of visualizations:

1. **Training History**: Shows the evolution of loss and parameters during training
2. **Fokker-Planck Terms**: Visualizes each term of the PDE and the residual
3. **Rating Forecasts**: Shows predicted rating trajectories with uncertainty
4. **Comparative Analysis**: Compares different players and different models
5. **Parameter Comparison**: Analyzes the inferred physical parameters across players

## Contributing

Contributions to improve the models, add new visualization techniques, or extend the analysis to different domains are welcome.

## License

This project is open source and available for academic and research purposes.
