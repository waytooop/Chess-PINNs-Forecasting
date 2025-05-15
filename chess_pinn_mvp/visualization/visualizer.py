"""
Visualization utilities for chess rating forecasting.

This module provides functions for creating various visualizations of rating data,
model predictions, and comparative analysis between multiple grandmasters.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_rating_history(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot historical rating data for a player.
    
    Args:
        df: DataFrame with rating data
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot rating history
    ax.plot(df.index, df['rating'], 'b-o', markersize=4, alpha=0.7)
    
    # Set title
    if title is None:
        title = "Rating History"
    ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel("Date")
    ax.set_ylabel("Rating")
    
    # Format x-axis to show dates properly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Add data point count
    ax.text(
        0.05, 0.95, f"Data points: {len(df)}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1)
    )
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Rating history plot saved to {save_path}")
    
    return fig


def plot_comparative_trajectories(
    gm_data: List[Dict[str, Any]],
    title: str = "Comparative Rating Trajectories",
    t_max: float = 365 * 5,  # 5 years
    n_points: int = 100,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparative rating trajectories for multiple grandmasters.
    
    Args:
        gm_data: List of dictionaries with 'model', 'name', 'r0', and 'color' keys
        title: Plot title
        t_max: Maximum time to forecast (in days)
        n_points: Number of time points
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Generate time points
    t_values = np.linspace(0, t_max, n_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot trajectories for each GM
    for data in gm_data:
        model = data['model']
        name = data['name']
        r0 = data['r0']
        color = data.get('color', None)
        
        # Generate forecast
        forecast = model.predict(r0=r0, t_values=t_values)
        
        # Plot mean forecast
        ax.plot(
            forecast['t'] / 365, forecast['mean'],
            label=f"{name} (start: {r0})",
            color=color
        )
        
        # Add uncertainty bands (lighter shade)
        ax.fill_between(
            forecast['t'] / 365, forecast['q25'], forecast['q75'],
            alpha=0.1, color=color
        )
        
        # Add starting point
        ax.scatter([0], [r0], color=color, s=50, zorder=5)
    
    # Set axis labels
    ax.set_xlabel("Years")
    ax.set_ylabel("Rating")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set title
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparative trajectories plot saved to {save_path}")
    
    return fig


def plot_rating_gains(
    gm_data: List[Dict[str, Any]],
    title: str = "Projected Rating Gains",
    t_max: float = 365 * 5,  # 5 years
    n_points: int = 100,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot projected rating gains for multiple grandmasters.
    
    Args:
        gm_data: List of dictionaries with 'model', 'name', 'r0', and 'color' keys
        title: Plot title
        t_max: Maximum time to forecast (in days)
        n_points: Number of time points
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Generate time points
    t_values = np.linspace(0, t_max, n_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Store final gains for annotation
    final_gains = []
    
    # Plot trajectories for each GM
    for data in gm_data:
        model = data['model']
        name = data['name']
        r0 = data['r0']
        color = data.get('color', None)
        
        # Generate forecast
        forecast = model.predict(r0=r0, t_values=t_values)
        
        # Calculate gains
        gains = forecast['mean'] - r0
        
        # Plot gains
        ax.plot(
            forecast['t'] / 365, gains,
            label=f"{name} (start: {r0})",
            color=color
        )
        
        # Add uncertainty bands (lighter shade)
        ax.fill_between(
            forecast['t'] / 365,
            forecast['q25'] - r0,
            forecast['q75'] - r0,
            alpha=0.1, color=color
        )
        
        # Store final gain
        final_gains.append((name, gains[-1], color))
    
    # Add zero line
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Set axis labels
    ax.set_xlabel("Years")
    ax.set_ylabel("Rating Gain")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set title
    ax.set_title(title)
    
    # Add annotations for final gains
    for name, gain, color in final_gains:
        ax.annotate(
            f"{gain:.1f}",
            xy=(t_values[-1] / 365, gain),
            xytext=(5, 0),
            textcoords='offset points',
            color=color,
            fontweight='bold'
        )
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Rating gains plot saved to {save_path}")
    
    return fig


def plot_milestone_achievement_rates(
    gm_data: List[Dict[str, Any]],
    title: str = "Milestone Achievement Rates",
    milestones: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot milestone achievement rates for multiple grandmasters.
    
    Args:
        gm_data: List of dictionaries with 'model', 'name', 'r0', and 'color' keys
        title: Plot title
        milestones: List of rating milestones to predict
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Default milestones if not provided
    if milestones is None:
        milestones = [2200, 2300, 2400, 2500, 2600, 2700, 2800, 2850, 2900]
    
    # Generate time points (20 years)
    t_values = np.linspace(0, 365 * 20, 1000)
    
    # Create dictionary to store milestone times for each GM
    milestone_data = {}
    
    # Get predictions for each GM
    for data in gm_data:
        model = data['model']
        name = data['name']
        r0 = data['r0']
        
        # Generate forecast
        forecast = model.predict(r0=r0, t_values=t_values)
        
        # Extract milestone times
        milestone_times = {}
        for milestone in milestones:
            # Only include milestones above the player's starting rating
            if milestone > r0:
                time = forecast['milestones'].get(milestone, float('inf'))
                if not np.isinf(time) and not np.isnan(time):
                    milestone_times[milestone] = time / 365  # Convert to years
        
        milestone_data[name] = milestone_times
    
    # Filter milestones that at least one player can achieve
    valid_milestones = set()
    for name, times in milestone_data.items():
        valid_milestones.update(times.keys())
    valid_milestones = sorted(valid_milestones)
    
    if not valid_milestones:
        logger.warning("No valid milestones found for any player")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar width
    n_players = len(gm_data)
    width = 0.8 / n_players
    
    # Plot bars for each GM
    for i, data in enumerate(gm_data):
        name = data['name']
        color = data.get('color', None)
        
        # Get times for this GM
        times = milestone_data[name]
        
        # Convert to arrays for plotting
        plot_milestones = []
        plot_times = []
        
        for milestone in valid_milestones:
            if milestone in times:
                plot_milestones.append(milestone)
                plot_times.append(times[milestone])
        
        # Calculate bar positions
        x = np.arange(len(plot_milestones))
        positions = x - 0.4 + (i + 0.5) * width
        
        # Plot bars
        ax.bar(
            positions, plot_times,
            width=width, label=name,
            color=color, alpha=0.7
        )
        
        # Add text labels
        for j, time in enumerate(plot_times):
            ax.text(
                positions[j], time + 0.1,
                f"{time:.1f}",
                ha='center', va='bottom',
                fontsize=8, rotation=90
            )
    
    # Set x-ticks at milestone positions
    ax.set_xticks(np.arange(len(valid_milestones)))
    ax.set_xticklabels(valid_milestones)
    
    # Set axis labels
    ax.set_xlabel("Rating Milestone")
    ax.set_ylabel("Years to Achieve")
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set title
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Milestone achievement plot saved to {save_path}")
    
    return fig


def plot_parameter_comparison(
    gm_data: List[Dict[str, Any]],
    title: str = "Parameter Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot parameter comparison for multiple grandmasters.
    
    Args:
        gm_data: List of dictionaries with 'model', 'name', and 'color' keys
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Get parameters for each GM
    param_data = []
    for data in gm_data:
        model = data['model']
        name = data['name']
        color = data.get('color', None)
        
        # Get parameters
        params = model.get_parameters()
        
        # Add to list
        param_data.append({
            'name': name,
            'color': color,
            'params': params
        })
    
    # Determine which parameters to plot
    is_nonlinear = any('gamma' in data['params'] for data in param_data)
    
    if is_nonlinear:
        # For nonlinear model
        param_names = ['alpha', 'gamma', 'beta', 'mu_eq']
        param_labels = ['α (mean-reversion)', 'γ (nonlinearity)', 'β (volatility decay)', 'μ_eq (equilibrium)']
        n_params = 4
    else:
        # For linear model
        param_names = ['alpha', 'sigma', 'mu_eq']
        param_labels = ['α (mean-reversion)', 'σ (volatility)', 'μ_eq (equilibrium)']
        n_params = 3
    
    # Create figure
    fig, axes = plt.subplots(1, n_params, figsize=(15, 6))
    
    # Plot bars for each parameter
    for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        ax = axes[i]
        
        # Special scaling for certain parameters
        scale_factor = 1.0
        if param_name == 'alpha':
            scale_factor = 1000.0  # Scale alpha by 1000
        elif param_name == 'beta' and is_nonlinear:
            scale_factor = 1000.0  # Scale beta by 1000
        
        # Get values and names
        values = [data['params'].get(param_name, 0) * scale_factor for data in param_data]
        names = [data['name'] for data in param_data]
        colors = [data['color'] for data in param_data]
        
        # Plot bars
        bars = ax.bar(
            np.arange(len(names)), values,
            color=colors, alpha=0.7
        )
        
        # Add value labels
        for j, value in enumerate(values):
            if param_name in ['alpha', 'beta']:
                label = f"{value:.3f}"
            elif param_name == 'gamma':
                label = f"{value:.2f}"
            else:
                label = f"{value:.0f}"
            
            ax.text(
                j, value + max(values) * 0.02,
                label,
                ha='center', va='bottom',
                fontsize=8, rotation=90
            )
        
        # Set title and labels
        if param_name == 'alpha':
            ax.set_title(f"{param_label} (×10³)")
        elif param_name == 'beta' and is_nonlinear:
            ax.set_title(f"{param_label} (×10³)")
        else:
            ax.set_title(param_label)
        
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
    
    # Set main title
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Parameter comparison plot saved to {save_path}")
    
    return fig


def plot_model_comparison(
    linear_model: Any,
    nonlinear_model: Any,
    r0: float,
    title: str = "Linear vs. Nonlinear Model Comparison",
    t_max: float = 365 * 5,  # 5 years
    n_points: int = 100,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison between linear and nonlinear models.
    
    Args:
        linear_model: Linear Fokker-Planck model
        nonlinear_model: Nonlinear Fokker-Planck model
        r0: Initial rating
        title: Plot title
        t_max: Maximum time to forecast (in days)
        n_points: Number of time points
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Generate time points
    t_values = np.linspace(0, t_max, n_points)
    
    # Generate forecasts
    linear_forecast = linear_model.predict(r0=r0, t_values=t_values)
    nonlinear_forecast = nonlinear_model.predict(r0=r0, t_values=t_values)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Mean forecast comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        linear_forecast['t'] / 365, linear_forecast['mean'],
        'b-', label='Linear Model'
    )
    ax1.plot(
        nonlinear_forecast['t'] / 365, nonlinear_forecast['mean'],
        'r-', label='Nonlinear Model'
    )
    ax1.scatter([0], [r0], color='k', s=50, zorder=5, label='Starting Rating')
    
    # Get parameters
    linear_params = linear_model.get_parameters()
    nonlinear_params = nonlinear_model.get_parameters()
    
    # Add equilibrium lines
    ax1.axhline(
        linear_params['mu_eq'], color='b', linestyle='--', alpha=0.5,
        label='Linear Equilibrium'
    )
    ax1.axhline(
        nonlinear_params['mu_eq'], color='r', linestyle='--', alpha=0.5,
        label='Nonlinear Equilibrium'
    )
    
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Rating')
    ax1.set_title('Mean Forecast Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Uncertainty comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot 50% confidence intervals
    ax2.fill_between(
        linear_forecast['t'] / 365,
        linear_forecast['q25'],
        linear_forecast['q75'],
        color='b', alpha=0.2, label='Linear 50% CI'
    )
    ax2.fill_between(
        nonlinear_forecast['t'] / 365,
        nonlinear_forecast['q25'],
        nonlinear_forecast['q75'],
        color='r', alpha=0.2, label='Nonlinear 50% CI'
    )
    
    # Plot mean forecasts (thinner lines)
    ax2.plot(
        linear_forecast['t'] / 365, linear_forecast['mean'],
        'b-', linewidth=1
    )
    ax2.plot(
        nonlinear_forecast['t'] / 365, nonlinear_forecast['mean'],
        'r-', linewidth=1
    )
    
    ax2.scatter([0], [r0], color='k', s=50, zorder=5)
    
    ax2.set_xlabel('Years')
    ax2.set_ylabel('Rating')
    ax2.set_title('Uncertainty Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rating gain comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate gains
    linear_gains = linear_forecast['mean'] - r0
    nonlinear_gains = nonlinear_forecast['mean'] - r0
    
    ax3.plot(
        linear_forecast['t'] / 365, linear_gains,
        'b-', label='Linear Model'
    )
    ax3.plot(
        nonlinear_forecast['t'] / 365, nonlinear_gains,
        'r-', label='Nonlinear Model'
    )
    
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Rating Gain')
    ax3.set_title('Rating Gain Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameters table
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Hide axes
    ax4.axis('off')
    
    # Create parameter comparison table
    param_table = [
        ['Parameter', 'Linear Model', 'Nonlinear Model'],
        ['α (mean-reversion)', f"{linear_params['alpha']:.6f}", f"{nonlinear_params['alpha']:.6f}"],
        ['μ_eq (equilibrium)', f"{linear_params['mu_eq']:.1f}", f"{nonlinear_params['mu_eq']:.1f}"]
    ]
    
    # Add linear model specific parameters
    if 'sigma' in linear_params:
        param_table.append(['σ (volatility)', f"{linear_params['sigma']:.2f}", "-"])
    
    # Add nonlinear model specific parameters
    if 'gamma' in nonlinear_params:
        param_table.append(['γ (nonlinearity)', "-", f"{nonlinear_params['gamma']:.2f}"])
    if 'beta' in nonlinear_params:
        param_table.append(['β (volatility decay)', "-", f"{nonlinear_params['beta']:.6f}"])
    if 'sigma0' in nonlinear_params:
        param_table.append(['σ₀ (base volatility)', "-", f"{nonlinear_params['sigma0']:.2f}"])
    
    # Create the table
    table = ax4.table(
        cellText=param_table,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3, 0.3]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Make header row bold
    for j in range(3):
        table[(0, j)].set_text_props(fontweight='bold')
    
    ax4.set_title('Model Parameters')
    
    # Add a common title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def create_summary_dashboard(
    gm_data: Dict[str, Any],
    output_dir: str,
    model_type: str = 'linear'
) -> None:
    """
    Create a summary dashboard for a grandmaster.
    
    Args:
        gm_data: Dictionary with GM data including:
            - df: DataFrame with rating history
            - model: Trained model
            - trainer: Model trainer
            - name: GM name
            - key: GM key
            - r0: Initial rating
        output_dir: Output directory
        model_type: Model type ('linear' or 'nonlinear')
    """
    # Extract data
    df = gm_data['df']
    model = gm_data['model']
    trainer = gm_data['trainer']
    name = gm_data['name']
    key = gm_data['key']
    r0 = gm_data['r0']
    
    # Create output directory
    dashboard_dir = os.path.join(output_dir, 'figures', f"{key}_{model_type}")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    logger.info(f"Creating dashboard for {name} ({model_type} model)")
    
    # 1. Plot rating history
    hist_path = os.path.join(dashboard_dir, f"{key}_history.png")
    plot_rating_history(
        df=df,
        title=f"{name}: Rating History",
        save_path=hist_path
    )
    
    # 2. Plot training history
    train_path = os.path.join(dashboard_dir, f"{key}_training_history.png")
    trainer.plot_training_history(save_path=train_path)
    
    # 3. Plot rating forecast
    forecast_path = os.path.join(dashboard_dir, f"{key}_forecast.png")
    trainer.plot_rating_forecast(
        r0=r0,
        save_path=forecast_path
    )
    
    # 4. Plot milestone predictions
    milestone_path = os.path.join(dashboard_dir, f"{key}_milestones.png")
    trainer.plot_milestone_predictions(
        r0=r0,
        milestones=[2400, 2500, 2600, 2700, 2800, 2850, 2900],
        save_path=milestone_path
    )
    
    logger.info(f"Dashboard for {name} ({model_type} model) created in {dashboard_dir}")
