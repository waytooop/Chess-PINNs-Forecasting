"""
Visualize the inverse problem for chess rating forecasting using Physics-Informed Neural Networks.

This script demonstrates how PINNs can infer physical parameters from sparse rating data,
and how these parameters are used to forecast future rating trajectories.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import project modules
from chess_pinn_mvp.utils.gm_config import GRANDMASTERS, GM_DICT
from chess_pinn_mvp.utils.data_processor import load_and_process_rating_data, RatingDataset, prepare_training_data
from chess_pinn_mvp.models.linear_fp_pinn import LinearFokkerPlanckPINN
from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from chess_pinn_mvp.models.trainer import FokkerPlanckTrainer
from chess_pinn_mvp.visualization.visualizer import (
    plot_rating_history, plot_comparative_trajectories, 
    plot_rating_gains, plot_parameter_comparison, plot_model_comparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
output_dir = Path("output/inverse_problem_visualization")
output_dir.mkdir(parents=True, exist_ok=True)

def visualize_fokker_planck_terms(
    model: torch.nn.Module,
    r0: float,
    t_max: float = 365 * 5,  # 5 years
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    n_points: int = 50,
    title: str = "Fokker-Planck PDE Terms",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize terms of the Fokker-Planck equation in a heatmap.
    
    Args:
        model: Trained PINN model
        r0: Initial rating
        t_max: Maximum time to visualize
        r_min: Minimum rating to visualize
        r_max: Maximum rating to visualize
        n_points: Number of points in each dimension
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set rating range if not provided
    if r_min is None:
        r_min = r0 - 200
    if r_max is None:
        r_max = r0 + 400
    
    # Create grid for visualization
    r_values = np.linspace(r_min, r_max, n_points)
    t_values = np.linspace(0, t_max, n_points)
    R, T = np.meshgrid(r_values, t_values)
    
    # Convert to tensors
    # Create tensors that properly track gradients
    r_flat = torch.tensor(R.flatten(), dtype=torch.float32, requires_grad=True)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32, requires_grad=True)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Get model parameters
    is_nonlinear = hasattr(model, 'gamma')
    params = model.get_parameters()
    
    # Compute derivatives - don't use no_grad since we need gradients
    derivs = model.compute_derivatives(r_flat, t_flat)
    
    # Extract numpy arrays from the derivatives
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
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.1])
    
    # Common args for all heatmaps
    heatmap_args = {
        'shading': 'gouraud',
        'alpha': 0.8,
        'cmap': 'viridis'
    }
    
    # 1. Probability density p(r,t)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(R, T / 365, p, **heatmap_args)
    ax1.set_title("Probability Density p(r,t)")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Time (years)")
    fig.colorbar(im1, ax=ax1)
    
    # 2. Time derivative ∂p/∂t
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(R, T / 365, p_t, **heatmap_args)
    ax2.set_title("Time Derivative ∂p/∂t")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Time (years)")
    fig.colorbar(im2, ax=ax2)
    
    # 3. Drift term -∂/∂r[μ(r)p]
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.pcolormesh(R, T / 365, drift_term, **heatmap_args)
    ax3.set_title("Drift Term -∂/∂r[μ(r)p]")
    ax3.set_xlabel("Rating")
    ax3.set_ylabel("Time (years)")
    fig.colorbar(im3, ax=ax3)
    
    # 4. Diffusion term (1/2)∂²/∂r²[σ²(r)p]
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.pcolormesh(R, T / 365, diffusion_term, **heatmap_args)
    ax4.set_title("Diffusion Term (1/2)∂²/∂r²[σ²(r)p]")
    ax4.set_xlabel("Rating")
    ax4.set_ylabel("Time (years)")
    fig.colorbar(im4, ax=ax4)
    
    # 5. Residual (PDE error)
    ax5 = fig.add_subplot(gs[2, :])
    im5 = ax5.pcolormesh(R, T / 365, residual, **heatmap_args)
    ax5.set_title("PDE Residual")
    ax5.set_xlabel("Rating")
    ax5.set_ylabel("Time (years)")
    fig.colorbar(im5, ax=ax5)
    
    # Add text with parameter values
    if is_nonlinear:
        param_text = (
            f"α={alpha:.5f}, γ={gamma:.2f}, σ₀={sigma0:.1f}, "
            f"β={beta:.5f}, μ_eq={mu_eq:.1f}, r₀={r0_ref:.1f}"
        )
    else:
        param_text = f"α={alpha:.5f}, σ={sigma:.1f}, μ_eq={mu_eq:.1f}"
    
    fig.text(0.5, 0.01, param_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set main title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Fokker-Planck terms visualization saved to {save_path}")
    
    return fig


def visualize_inverse_problem_training(
    trainer: FokkerPlanckTrainer,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create an enhanced visualization of the inverse problem training process.
    This shows how parameters are inferred from data during training.
    
    Args:
        trainer: Trained FokkerPlanckTrainer
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Extract training history
    history = trainer.history
    
    # Check if model is nonlinear
    is_nonlinear = hasattr(trainer.model, 'gamma')
    
    # Create figure with more detailed subplots
    fig = plt.figure(figsize=(15, 12))
    
    if is_nonlinear:
        gs = fig.add_gridspec(3, 2)
    else:
        gs = fig.add_gridspec(2, 2)
    
    # 1. Training vs. Validation Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['epochs'], history['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs. Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Components
    ax2 = fig.add_subplot(gs[0, 1])
    if 'data_loss' in history and 'physics_loss' in history:
        ax2.plot(history['epochs'], history['data_loss'], 'g-', label='Data Loss')
        ax2.plot(history['epochs'], history['physics_loss'], 'm-', label='Physics Loss')
        ax2.set_yscale('log')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Component Loss')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Parameter Evolution - Mean Reversion
    ax3 = fig.add_subplot(gs[1, 0])
    if 'alpha' in history:
        ax3.plot(history['epochs'], history['alpha'], 'b-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('α')
        ax3.set_title('Mean-Reversion Parameter (α)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Other parameters
    if is_nonlinear:
        # For nonlinear model, show gamma, beta, and equilibrium
        ax4 = fig.add_subplot(gs[1, 1])
        if 'gamma' in history:
            ax4.plot(history['epochs'], history['gamma'], 'r-', label='γ')
            if 'beta' in history:
                # Scale beta to make it visible on same scale
                ax4.plot(history['epochs'], np.array(history['beta']) * 1000, 'g-', label='β×10³')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Parameter Value')
            ax4.set_title('Nonlinearity and Volatility Decay')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Base volatility
        ax5 = fig.add_subplot(gs[2, 0])
        if 'sigma0' in history:
            ax5.plot(history['epochs'], history['sigma0'], 'm-')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('σ₀')
            ax5.set_title('Base Volatility (σ₀)')
            ax5.grid(True, alpha=0.3)
        
        # Equilibrium rating
        ax6 = fig.add_subplot(gs[2, 1])
        if 'mu_eq' in history:
            ax6.plot(history['epochs'], history['mu_eq'], 'c-')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('μ_eq')
            ax6.set_title('Equilibrium Rating (μ_eq)')
            ax6.grid(True, alpha=0.3)
    else:
        # For linear model, show volatility and equilibrium
        ax4 = fig.add_subplot(gs[1, 1])
        if 'sigma' in history:
            ax4.plot(history['epochs'], history['sigma'], 'r-')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('σ')
            ax4.set_title('Volatility (σ)')
            ax4.grid(True, alpha=0.3)
    
    # Set main title
    model_type = "Nonlinear 'Chess Asymptotic'" if is_nonlinear else "Linear Ornstein-Uhlenbeck"
    fig.suptitle(f'Inverse Problem Training History - {model_type} Model', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Inverse problem training visualization saved to {save_path}")
    
    return fig


def main():
    """Main execution function."""
    logger.info("Starting inverse problem visualization")
    
    # Select grandmasters to analyze
    selected_gms = ["GM001", "GM002", "GM003", "GM005", "GM006", "GM007"]
    
    # Training parameters
    train_params = {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'physics_weight': 0.7,
        'collocation_points': 1000,
        'val_split': 0.2
    }
    
    # Linear model hyper-parameters
    linear_params = {
        'hidden_layers': [64, 128, 128, 64],
        'activation': 'tanh',
        'initial_alpha': 0.05,
        'initial_sigma': 100.0,
        'initial_mu_eq': 2800.0
    }
    
    # Nonlinear model hyper-parameters
    nonlinear_params = {
        'hidden_layers': [64, 128, 128, 64],
        'activation': 'tanh',
        'initial_alpha': 0.05,
        'initial_gamma': 2.0,
        'initial_sigma0': 100.0,
        'initial_beta': 0.005,
        'initial_mu_eq': 2800.0,
        'initial_r0': 2200.0
    }
    
    # Containers for trained models and data
    linear_models = {}
    nonlinear_models = {}
    linear_trainers = {}
    nonlinear_trainers = {}
    gm_data = {}
    
    # Process each GM
    for gm_key in selected_gms:
        gm_info = GM_DICT[gm_key]
        gm_name = gm_info['name']
        fide_id = gm_info['id']
        
        logger.info(f"Processing {gm_name} (FIDE ID: {fide_id})")
        
        # Load and process rating data
        try:
            df = load_and_process_rating_data(fide_id)
            
            if df is None or len(df) < 10:
                logger.warning(f"Insufficient data for {gm_name}. Skipping.")
                continue
            
            # Store data
            gm_data[gm_key] = {
                'name': gm_name,
                'df': df,
                'r0': df['rating'].iloc[0]  # Initial rating
            }
            
            # Train linear model
            logger.info(f"Training linear model for {gm_name}")
            linear_model = LinearFokkerPlanckPINN(**linear_params)
            linear_trainer = FokkerPlanckTrainer(
                model=linear_model,
                lr=train_params['learning_rate'],
                device=device
            )
            
            # Create RatingDataset from DataFrame
            train_data, val_data = prepare_training_data(
                rating_data=df,
                test_fraction=train_params['val_split'],
                validate_data=True
            )
            
            if train_data is None:
                raise ValueError(f"Failed to prepare training data for {gm_name}")
                
            # Set physics weight
            linear_trainer.physics_weight = train_params['physics_weight']
            
            # Create datasets
            train_dataset = RatingDataset(train_data)
            val_dataset = RatingDataset(val_data) if val_data is not None else None
            
            # Train the model
            linear_trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=train_params['epochs'],
                batch_size=train_params['batch_size'],
                collocation_points=train_params['collocation_points'],
                verbose=True
            )
            
            # Store trained model and trainer
            linear_models[gm_key] = linear_model
            linear_trainers[gm_key] = linear_trainer
            
            # Train nonlinear model
            logger.info(f"Training nonlinear model for {gm_name}")
            nonlinear_model = NonlinearFokkerPlanckPINN(**nonlinear_params)
            nonlinear_trainer = FokkerPlanckTrainer(
                model=nonlinear_model,
                lr=train_params['learning_rate'],
                device=device
            )
            
            # Set physics weight
            nonlinear_trainer.physics_weight = train_params['physics_weight']
            
            # Train the model (reusing datasets from linear model)
            nonlinear_trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=train_params['epochs'],
                batch_size=train_params['batch_size'],
                collocation_points=train_params['collocation_points'],
                verbose=True
            )
            
            # Store trained model and trainer
            nonlinear_models[gm_key] = nonlinear_model
            nonlinear_trainers[gm_key] = nonlinear_trainer
            
        except Exception as e:
            logger.error(f"Error processing {gm_name}: {str(e)}")
    
    # Create visualizations
    logger.info("Creating visualizations")
    
    # 1. Create individual GM dashboards
    for gm_key, gm_info in gm_data.items():
        gm_name = gm_info['name']
        df = gm_info['df']
        r0 = gm_info['r0']
        
        logger.info(f"Creating dashboard for {gm_name}")
        
        # Linear model visualizations
        if gm_key in linear_trainers:
            # Training history
            linear_trainer = linear_trainers[gm_key]
            linear_model = linear_models[gm_key]
            
            # Create directory
            gm_dir = output_dir / f"{gm_key}_linear"
            gm_dir.mkdir(exist_ok=True)
            
            # Training history
            visualize_inverse_problem_training(
                trainer=linear_trainer,
                save_path=gm_dir / f"{gm_key}_training_history.png"
            )
            
            # Rating forecast
            linear_trainer.plot_rating_forecast(
                r0=r0,
                save_path=gm_dir / f"{gm_key}_forecast.png"
            )
            
            # Fokker-Planck terms
            visualize_fokker_planck_terms(
                model=linear_model,
                r0=r0,
                title=f"{gm_name} - Linear Model - Fokker-Planck Terms",
                save_path=gm_dir / f"{gm_key}_residuals.png"
            )
        
        # Nonlinear model visualizations
        if gm_key in nonlinear_trainers:
            # Training history
            nonlinear_trainer = nonlinear_trainers[gm_key]
            nonlinear_model = nonlinear_models[gm_key]
            
            # Create directory
            gm_dir = output_dir / f"{gm_key}_nonlinear"
            gm_dir.mkdir(exist_ok=True)
            
            # Training history
            visualize_inverse_problem_training(
                trainer=nonlinear_trainer,
                save_path=gm_dir / f"{gm_key}_nonlinear_training_history.png"
            )
            
            # Rating forecast
            nonlinear_trainer.plot_rating_forecast(
                r0=r0,
                save_path=gm_dir / f"{gm_key}_nonlinear_forecast.png"
            )
            
            # Fokker-Planck terms
            visualize_fokker_planck_terms(
                model=nonlinear_model,
                r0=r0,
                title=f"{gm_name} - Nonlinear Model - Fokker-Planck Terms",
                save_path=gm_dir / f"{gm_key}_nonlinear_residuals.png"
            )
            
            # Model comparison
            if gm_key in linear_models:
                plot_model_comparison(
                    linear_model=linear_models[gm_key],
                    nonlinear_model=nonlinear_models[gm_key],
                    r0=r0,
                    title=f"{gm_name} - Linear vs. Nonlinear Model Comparison",
                    save_path=gm_dir / f"{gm_key}_model_comparison.png"
                )
    
    # 2. Comparative analysis across GMs
    if len(linear_models) >= 2:
        # Create comparative data structure
        linear_gm_data = []
        nonlinear_gm_data = []
        
        # Define colors for each GM
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (gm_key, gm_info) in enumerate(gm_data.items()):
            if gm_key in linear_models:
                color = colors[i % len(colors)]
                
                # Linear model data
                linear_gm_data.append({
                    'model': linear_models[gm_key],
                    'name': gm_info['name'],
                    'r0': gm_info['r0'],
                    'color': color
                })
                
                # Nonlinear model data
                if gm_key in nonlinear_models:
                    nonlinear_gm_data.append({
                        'model': nonlinear_models[gm_key],
                        'name': gm_info['name'],
                        'r0': gm_info['r0'],
                        'color': color
                    })
        
        # Create comparative directory
        comp_dir = output_dir / "comparative"
        comp_dir.mkdir(exist_ok=True)
        
        # Comparative trajectories - Linear
        plot_comparative_trajectories(
            gm_data=linear_gm_data,
            title="Comparative Rating Trajectories - Linear Model",
            save_path=comp_dir / "comparative_trajectories_standard.png"
        )
        
        # Comparative trajectories - Nonlinear
        plot_comparative_trajectories(
            gm_data=nonlinear_gm_data,
            title="Comparative Rating Trajectories - Nonlinear Model",
            save_path=comp_dir / "comparative_trajectories_nonlinear_standard.png"
        )
        
        # Rating gains - Linear
        plot_rating_gains(
            gm_data=linear_gm_data,
            title="Rating Gains Comparison - Linear Model",
            save_path=comp_dir / "rating_gains_standard.png"
        )
        
        # Rating gains - Nonlinear
        plot_rating_gains(
            gm_data=nonlinear_gm_data,
            title="Rating Gains Comparison - Nonlinear Model",
            save_path=comp_dir / "rating_gains_nonlinear_standard.png"
        )
        
        # Parameter comparison - Linear
        plot_parameter_comparison(
            gm_data=linear_gm_data,
            title="Parameter Comparison - Linear Model",
            save_path=comp_dir / "parameter_comparison_linear.png"
        )
        
        # Parameter comparison - Nonlinear
        plot_parameter_comparison(
            gm_data=nonlinear_gm_data,
            title="Parameter Comparison - Nonlinear Model",
            save_path=comp_dir / "parameter_comparison_nonlinear.png"
        )
    
    logger.info(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
