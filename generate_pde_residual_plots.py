"""
Generate high-quality PDE residual plots for the Chess Rating forecasting research paper.

This script creates detailed visualizations of the Fokker-Planck equation terms
for both the linear and nonlinear models, optimized for publication quality.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, Optional, Any

# Import project modules
from chess_pinn_mvp.utils.gm_config import GRANDMASTERS, GM_DICT
from chess_pinn_mvp.utils.data_processor import load_and_process_rating_data
from chess_pinn_mvp.models.linear_fp_pinn import LinearFokkerPlanckPINN
from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from chess_pinn_mvp.models.trainer import FokkerPlanckTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
output_dir = Path("output/pde_residual_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Set matplotlib style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name for newer matplotlib
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (10, 8),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})


def generate_publication_residual_plot(
    model: torch.nn.Module,
    r0: float,
    t_max: float = 365 * 5,  # 5 years
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    n_points: int = 100,  # Higher resolution
    title: str = "Fokker-Planck PDE Terms",
    gm_name: str = "",
    model_type: str = "linear",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate publication-quality visualization of the Fokker-Planck equation terms.
    
    Args:
        model: Trained PINN model
        r0: Initial rating
        t_max: Maximum time to visualize
        r_min: Minimum rating to visualize
        r_max: Maximum rating to visualize
        n_points: Number of points in each dimension
        title: Plot title
        gm_name: Grandmaster name
        model_type: Model type ('linear' or 'nonlinear')
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set rating range if not provided
    if r_min is None:
        r_min = r0 - 200
    if r_max is None:
        r_max = r0 + 400
    
    # Create grid for visualization with higher resolution
    r_values = np.linspace(r_min, r_max, n_points)
    t_values = np.linspace(0, t_max, n_points)
    R, T = np.meshgrid(r_values, t_values)
    
    # Convert to tensors
    r_flat = torch.tensor(R.flatten(), dtype=torch.float32, requires_grad=True)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32, requires_grad=True)
    
    # Get model parameters
    is_nonlinear = hasattr(model, 'gamma')
    params = model.get_parameters()
    
    # Ensure model is in eval mode
    model.eval()
    
    # Compute derivatives - need gradients for automatic differentiation
    derivs = model.compute_derivatives(r_flat, t_flat)
    
    # Extract numpy arrays from derivatives
    with torch.no_grad():
        p = derivs['p'].detach().numpy().reshape(R.shape)
        p_t = derivs['p_t'].detach().numpy().reshape(R.shape)
        p_r = derivs['p_r'].detach().numpy().reshape(R.shape)
        p_rr = derivs['p_rr'].detach().numpy().reshape(R.shape)
        
        # Compute drift term
        if is_nonlinear:
            # Nonlinear drift: μ(r) = -α(r-μ_eq)^γ
            alpha = params['alpha']
            gamma = params['gamma']
            mu_eq = params['mu_eq']
            
            r_diff = R - mu_eq
            sign = np.sign(r_diff)
            abs_diff = np.abs(r_diff)
            r_diff_pow = sign * (abs_diff ** gamma)
            mu = -alpha * r_diff_pow
            
            # Derivative of drift
            mu_r = -alpha * gamma * sign * (abs_diff ** (gamma - 1))
            
            # State-dependent diffusion
            sigma0 = params['sigma0']
            beta = params['beta']
            r0_ref = params['r0']
            
            sigma_squared = sigma0**2 * np.exp(-beta * (R - r0_ref))
            sigma_squared_r = -beta * sigma_squared
            
            # Drift term
            drift_term = mu * p_r + p * mu_r
            
            # Diffusion term
            diffusion_term = 0.5 * (sigma_squared * p_rr + sigma_squared_r * p_r)
        else:
            # Linear drift: μ(r) = -α(r-μ_eq)
            alpha = params['alpha']
            mu_eq = params['mu_eq']
            mu = -alpha * (R - mu_eq)
            mu_r = -alpha * np.ones_like(R)
            
            # Constant diffusion
            sigma = params['sigma']
            sigma_squared = sigma**2
            
            # Drift term
            drift_term = mu * p_r + p * mu_r
            
            # Diffusion term
            diffusion_term = 0.5 * sigma_squared * p_rr
        
        # Residual
        residual = p_t + drift_term - diffusion_term
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.1])
    
    # Custom colormap for better visual distinction
    cmap_p = 'viridis'         # For probability density
    cmap_deriv = 'coolwarm'    # For derivatives (diverging)
    cmap_res = 'RdBu_r'        # For residual (diverging)
    
    # Common args for all heatmaps
    heatmap_args = {
        'shading': 'gouraud',
        'alpha': 0.8,
    }
    
    # 1. Probability density p(r,t)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(R, T / 365, p, cmap=cmap_p, **heatmap_args)
    ax1.set_title("Probability Density $p(r,t)$")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Time (years)")
    fig.colorbar(im1, ax=ax1)
    
    # 2. Time derivative ∂p/∂t
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(R, T / 365, p_t, cmap=cmap_deriv, **heatmap_args)
    ax2.set_title(r"Time Derivative $\frac{\partial p}{\partial t}$")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Time (years)")
    fig.colorbar(im2, ax=ax2)
    
    # 3. Drift term -∂/∂r[μ(r)p]
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.pcolormesh(R, T / 365, drift_term, cmap=cmap_deriv, **heatmap_args)
    
    # Use LaTeX for equation
    if is_nonlinear:
        drift_eq = r"Drift Term $-\frac{\partial}{\partial r}[-\alpha(r-\mu_{eq})^{\gamma}p]$"
    else:
        drift_eq = r"Drift Term $-\frac{\partial}{\partial r}[-\alpha(r-\mu_{eq})p]$"
    
    ax3.set_title(drift_eq)
    ax3.set_xlabel("Rating")
    ax3.set_ylabel("Time (years)")
    fig.colorbar(im3, ax=ax3)
    
    # 4. Diffusion term (1/2)∂²/∂r²[σ²(r)p]
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.pcolormesh(R, T / 365, diffusion_term, cmap=cmap_deriv, **heatmap_args)
    
    # Use LaTeX for equation
    if is_nonlinear:
        diff_eq = r"Diffusion Term $\frac{1}{2}\frac{\partial^2}{\partial r^2}[\sigma_0^2 e^{-\beta(r-r_0)}p]$"
    else:
        diff_eq = r"Diffusion Term $\frac{1}{2}\frac{\partial^2}{\partial r^2}[\sigma^2 p]$"
    
    ax4.set_title(diff_eq)
    ax4.set_xlabel("Rating")
    ax4.set_ylabel("Time (years)")
    fig.colorbar(im4, ax=ax4)
    
    # 5. Residual (PDE error)
    ax5 = fig.add_subplot(gs[2, :])
    im5 = ax5.pcolormesh(R, T / 365, residual, cmap=cmap_res, **heatmap_args)
    ax5.set_title("PDE Residual")
    ax5.set_xlabel("Rating")
    ax5.set_ylabel("Time (years)")
    fig.colorbar(im5, ax=ax5)
    
    # Add text with parameter values
    if is_nonlinear:
        param_text = (
            f"$\\alpha={alpha:.5f}$, $\\gamma={gamma:.2f}$, $\\sigma_0={sigma0:.1f}$, "
            f"$\\beta={beta:.5f}$, $\\mu_{{eq}}={mu_eq:.1f}$, $r_0={r0_ref:.1f}$"
        )
    else:
        param_text = f"$\\alpha={alpha:.5f}$, $\\sigma={sigma:.1f}$, $\\mu_{{eq}}={mu_eq:.1f}$"
    
    fig.text(0.5, 0.01, param_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set main title
    if gm_name:
        model_name = "Nonlinear 'Chess Asymptotic'" if is_nonlinear else "Linear Ornstein-Uhlenbeck"
        fig.suptitle(f"{gm_name} - {model_name} Model", fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Fokker-Planck terms visualization saved to {save_path}")
    
    return fig


def train_gm_model(
    gm_key: str,
    epochs: int = 100,
    batch_size: int = 16,
    physics_weight: float = 0.7,
    is_nonlinear: bool = False
) -> tuple:
    """
    Train a model for a specific grandmaster.
    
    Args:
        gm_key: GM key from gm_config
        epochs: Number of training epochs
        batch_size: Training batch size
        physics_weight: Weight for physics loss
        is_nonlinear: Whether to use nonlinear model
        
    Returns:
        Tuple of (model, trainer, df, r0)
    """
    # Get GM info
    gm_info = GM_DICT[gm_key]
    gm_name = gm_info['name']
    fide_id = gm_info['id']
    
    logger.info(f"Processing {gm_name} (FIDE ID: {fide_id})")
    
    # Load and process rating data
    df = load_and_process_rating_data(fide_id)
    
    if df is None or len(df) < 10:
        logger.warning(f"Insufficient data for {gm_name}. Skipping.")
        return None, None, None, None
    
    # Get initial rating
    r0 = df['rating'].iloc[0]
    
    # Define model parameters
    if is_nonlinear:
        # Nonlinear model
        model = NonlinearFokkerPlanckPINN(
            hidden_layers=[64, 128, 128, 64],
            activation='tanh',
            initial_alpha=0.05,
            initial_gamma=2.0,
            initial_sigma0=100.0,
            initial_beta=0.005,
            initial_mu_eq=2800.0,
            initial_r0=2200.0
        )
    else:
        # Linear model
        model = LinearFokkerPlanckPINN(
            hidden_layers=[64, 128, 128, 64],
            activation='tanh',
            initial_alpha=0.05,
            initial_sigma=100.0,
            initial_mu_eq=2800.0
        )
    
    # Create trainer
    trainer = FokkerPlanckTrainer(
        model=model,
        lr=0.001,
        device=device
    )
    
    # Prepare training data
    from chess_pinn_mvp.utils.data_processor import RatingDataset, prepare_training_data
    
    # Create training datasets
    train_data, val_data = prepare_training_data(
        rating_data=df,
        test_fraction=0.2,
        validate_data=True
    )
    
    if train_data is None:
        logger.warning(f"Failed to prepare training data for {gm_key}")
        return None, None, None, None
    
    # Set physics weight
    trainer.physics_weight = physics_weight
    
    # Create datasets
    train_dataset = RatingDataset(train_data)
    val_dataset = RatingDataset(val_data) if val_data is not None else None
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        collocation_points=1000,
        verbose=True
    )
    
    return model, trainer, df, r0


def main():
    """Main execution function."""
    logger.info("Starting PDE residual plot generation")
    
    # Selected GMs for analysis
    selected_gms = {
        "GM001": "Magnus Carlsen",  # World Champion
        "GM003": "Gukesh D",        # Rising star
        "GM005": "Alireza Firouzja" # Young prodigy
    }
    
    # Training parameters
    epochs = 150  # More epochs for better convergence
    batch_size = 16
    physics_weight = 0.7
    
    # Process each GM
    for gm_key, gm_name in selected_gms.items():
        logger.info(f"Processing {gm_name}")
        
        # Create GM directory
        gm_dir = output_dir / gm_key
        gm_dir.mkdir(exist_ok=True)
        
        # Train and generate plots for linear model
        logger.info(f"Training linear model for {gm_name}")
        linear_model, linear_trainer, df, r0 = train_gm_model(
            gm_key=gm_key,
            epochs=epochs,
            batch_size=batch_size,
            physics_weight=physics_weight,
            is_nonlinear=False
        )
        
        if linear_model is not None:
            # Generate linear model residual plot
            generate_publication_residual_plot(
                model=linear_model,
                r0=r0,
                gm_name=gm_name,
                model_type="linear",
                save_path=gm_dir / f"{gm_key}_residuals.png"
            )
        
        # Train and generate plots for nonlinear model
        logger.info(f"Training nonlinear model for {gm_name}")
        nonlinear_model, nonlinear_trainer, df, r0 = train_gm_model(
            gm_key=gm_key,
            epochs=epochs,
            batch_size=batch_size,
            physics_weight=physics_weight,
            is_nonlinear=True
        )
        
        if nonlinear_model is not None:
            # Generate nonlinear model residual plot
            generate_publication_residual_plot(
                model=nonlinear_model,
                r0=r0,
                gm_name=gm_name,
                model_type="nonlinear",
                save_path=gm_dir / f"{gm_key}_nonlinear_residuals.png"
            )
    
    logger.info(f"All PDE residual plots saved to {output_dir}")

if __name__ == "__main__":
    main()
