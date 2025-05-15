"""
Test script to compare the original nonlinear model with the improved version.

This script demonstrates the differences in forecasting behavior between the two models,
particularly focusing on the stability of rating predictions.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any

# Import the models
from chess_pinn_mvp.models.nonlinear_fp_pinn import NonlinearFokkerPlanckPINN
from improved_nonlinear_fp_pinn import ImprovedNonlinearFokkerPlanckPINN

# Create output directory
os.makedirs('output/model_comparison', exist_ok=True)

def compare_models(r0: float, model_params: Dict[str, Any] = None):
    """
    Compare original and improved nonlinear models for a given starting rating.
    
    Args:
        r0: Initial rating
        model_params: Parameters to use for both models
    """
    # Default parameters if none provided
    if model_params is None:
        model_params = {
            'hidden_layers': [64, 128, 128, 64],
            'activation': 'tanh',
            'initial_alpha': 0.05,
            'initial_gamma': 2.0,
            'initial_sigma0': 100.0,
            'initial_beta': 0.005,
            'initial_mu_eq': 2800.0,
            'initial_r0': 2200.0
        }
    
    # Create models
    original_model = NonlinearFokkerPlanckPINN(**model_params)
    
    # Add improved model parameters
    improved_params = model_params.copy()
    improved_params.update({
        'min_rating_fraction': 0.9,
        'stabilization_factor': 0.5
    })
    improved_model = ImprovedNonlinearFokkerPlanckPINN(**improved_params)
    
    # Generate time points (5 years)
    t_values = np.linspace(0, 365 * 5, 100)
    
    # Generate predictions
    orig_pred = original_model.predict(r0=r0, t_values=t_values)
    impr_pred = improved_model.predict(r0=r0, t_values=t_values)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot original model
    ax1.plot(orig_pred['t'] / 365, orig_pred['mean'], 'r-', label='Mean Forecast')
    ax1.fill_between(orig_pred['t'] / 365, orig_pred['q25'], orig_pred['q75'], 
                    color='r', alpha=0.2, label='50% CI')
    ax1.scatter([0], [r0], color='k', s=50, zorder=5, label='Starting Rating')
    ax1.axhline(model_params['initial_mu_eq'], color='r', linestyle='--', alpha=0.5,
               label='Equilibrium')
    
    ax1.set_title(f"Original Nonlinear Model (r0={r0})")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Rating")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot improved model
    ax2.plot(impr_pred['t'] / 365, impr_pred['mean'], 'g-', label='Mean Forecast')
    ax2.fill_between(impr_pred['t'] / 365, impr_pred['q25'], impr_pred['q75'], 
                    color='g', alpha=0.2, label='50% CI')
    ax2.scatter([0], [r0], color='k', s=50, zorder=5, label='Starting Rating')
    ax2.axhline(model_params['initial_mu_eq'], color='g', linestyle='--', alpha=0.5,
               label='Equilibrium')
    
    ax2.set_title(f"Improved Nonlinear Model (r0={r0})")
    ax2.set_xlabel("Years")
    ax2.set_ylabel("Rating")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set consistent y-axis limits for better comparison
    y_min = min(orig_pred['mean'].min(), impr_pred['mean'].min()) - 100
    y_max = max(orig_pred['mean'].max(), impr_pred['mean'].max()) + 100
    
    # Ensure y_min isn't unreasonably low - set minimum to at least 0.9 * r0
    y_min = max(y_min, 0.9 * r0 - 100)
    
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.suptitle(f"Rating Forecast Comparison - Starting Rating {r0}", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    save_path = f'output/model_comparison/comparison_r0_{int(r0)}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison to {save_path}")
    
    return fig

def main():
    """Main function to run model comparisons."""
    
    # Define standard model parameters
    model_params = {
        'hidden_layers': [64, 128, 128, 64],
        'activation': 'tanh',
        'initial_alpha': 0.05,
        'initial_gamma': 2.0,
        'initial_sigma0': 100.0,
        'initial_beta': 0.005,
        'initial_mu_eq': 2800.0,
        'initial_r0': 2200.0
    }
    
    # Test with different starting ratings
    starting_ratings = [2400, 2200]
    
    for r0 in starting_ratings:
        compare_models(r0, model_params)
    
    print("All comparisons complete.")

if __name__ == "__main__":
    main()
