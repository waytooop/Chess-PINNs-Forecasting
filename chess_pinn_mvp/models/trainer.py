"""
PINN Trainer class for chess rating forecasting models.

This module provides the training infrastructure for both linear and nonlinear
Fokker-Planck PINN models, handling:
1. Data and physics loss calculation
2. Optimization
3. Training loop with early stopping
4. Model saving and loading
5. Prediction and visualization utilities
"""
import os
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from chess_pinn_mvp.utils.data_processor import RatingDataset, batch_to_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FokkerPlanckTrainer:
    """
    Trainer for Fokker-Planck PINN models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        physics_weight: float = 0.5,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PINN model to train
            lr: Learning rate for optimizer
            physics_weight: Weight for physics loss (0 to 1)
            device: Device to use for training
            output_dir: Directory to save outputs
            model_name: Name for saved model files
        """
        self.model = model
        self.lr = lr
        self.physics_weight = physics_weight
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Set output directory
        self.output_dir = output_dir if output_dir is not None else './output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model name
        self.model_name = model_name if model_name is not None else f"fp_pinn_{int(time.time())}"
        
        # Initialize training history
        self.history = {
            'epochs': [],
            'loss': [],
            'data_loss': [],
            'physics_loss': [],
            'val_loss': [],
            'val_data_loss': [],
            'val_physics_loss': [],
            'alpha': [],
            'sigma': [],
            'mu_eq': [],
            # Additional parameters for nonlinear model
            'gamma': [],
            'beta': [],
            'r0': []
        }
        
        # Create subdirectories
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.figs_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)
        
        logger.info(f"Initialized trainer for {model.__class__.__name__} on {self.device}")
    
    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        collocation_points: int = 1000,
        return_components: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the training loss for a batch.
        
        Args:
            batch: Dictionary with 't', 'r', and 'r_original' tensors
            collocation_points: Number of points to use for PDE residual
            return_components: Whether to return loss components
            
        Returns:
            Total loss if return_components is False, otherwise a tuple of
            (total_loss, {'data_loss': data_loss, 'physics_loss': physics_loss})
        """
        # Move data to device
        batch = batch_to_device(batch, self.device)
        
        # Extract tensors
        t = batch['t']
        r = batch['r']
        
        # Generate collocation points for PDE residual
        t_physics = torch.rand(
            collocation_points, device=self.device
        ) * torch.max(t).item()
        t_physics.requires_grad_(True)
        
        r_physics = torch.normal(
            mean=r.mean(),
            std=r.std(),
            size=(collocation_points,),
            device=self.device
        )
        r_physics.requires_grad_(True)
        
        # Data loss (negative log-likelihood)
        p_pred = self.model(r, t)
        
        # Ensure the PDF integrates to approximately 1
        # We do this by computing the normalization factor using trapezoidal rule
        # This is a simplified approach; a more accurate approach would define a grid
        r_sorted, indices = torch.sort(r)
        p_sorted = p_pred[indices].squeeze()
        normalization_factor = torch.trapezoid(p_sorted, r_sorted)
        p_pred_normalized = p_pred / (normalization_factor + 1e-8)
        
        data_loss = -torch.log(p_pred_normalized + 1e-8).mean()
        
        # Physics loss (PDE residual)
        pde_residual = self.model.fokker_planck_residual(r_physics, t_physics)
        physics_loss = torch.mean(pde_residual**2)
        
        # Combined loss with weighting
        total_loss = (1 - self.physics_weight) * data_loss + self.physics_weight * physics_loss
        
        if return_components:
            return total_loss, {
                'data_loss': data_loss.item(),
                'physics_loss': physics_loss.item()
            }
        else:
            return total_loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        collocation_points: int = 1000
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            collocation_points: Number of collocation points for PDE residual
            
        Returns:
            Dictionary with training metrics
        """
        # Training mode
        self.model.train()
        
        # Initialize metrics
        epoch_metrics = {
            'loss': 0.0,
            'data_loss': 0.0,
            'physics_loss': 0.0,
            'val_loss': 0.0,
            'val_data_loss': 0.0,
            'val_physics_loss': 0.0
        }
        
        # Training loop
        for batch in train_loader:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, loss_components = self.compute_losses(
                batch=batch,
                collocation_points=collocation_points,
                return_components=True
            )
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['loss'] += loss.item() / len(train_loader)
            epoch_metrics['data_loss'] += loss_components['data_loss'] / len(train_loader)
            epoch_metrics['physics_loss'] += loss_components['physics_loss'] / len(train_loader)
        
        # Validation loop
        if val_loader is not None:
            self.model.eval()
            # Don't use torch.no_grad() since we need gradients for PDE residuals
            for batch in val_loader:
                # Compute validation loss
                val_loss, val_loss_components = self.compute_losses(
                    batch=batch,
                    collocation_points=collocation_points,
                    return_components=True
                )
                
                # Update metrics
                epoch_metrics['val_loss'] += val_loss.item() / len(val_loader)
                epoch_metrics['val_data_loss'] += val_loss_components['data_loss'] / len(val_loader)
                epoch_metrics['val_physics_loss'] += val_loss_components['physics_loss'] / len(val_loader)
        
        # Get current physics parameters
        params = self.model.get_parameters()
        for key, value in params.items():
            epoch_metrics[key] = value
        
        return epoch_metrics
    
    def train(
        self,
        train_dataset: RatingDataset,
        val_dataset: Optional[RatingDataset] = None,
        epochs: int = 100,
        batch_size: int = 32,
        collocation_points: int = 1000,
        patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            collocation_points: Number of collocation points for PDE residual
            patience: Patience for early stopping
            verbose: Whether to log training progress
            
        Returns:
            Training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Initialize best loss for early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Train one epoch
            epoch_metrics = self.train_epoch(
                train_loader=train_loader,
                val_loader=val_loader,
                collocation_points=collocation_points
            )
            
            # Update history
            self.history['epochs'].append(epoch + 1)
            self.history['loss'].append(epoch_metrics['loss'])
            self.history['data_loss'].append(epoch_metrics['data_loss'])
            self.history['physics_loss'].append(epoch_metrics['physics_loss'])
            self.history['val_loss'].append(epoch_metrics.get('val_loss', 0.0))
            self.history['val_data_loss'].append(epoch_metrics.get('val_data_loss', 0.0))
            self.history['val_physics_loss'].append(epoch_metrics.get('val_physics_loss', 0.0))
            
            # Update parameter history
            self.history['alpha'].append(epoch_metrics.get('alpha', 0.0))
            self.history['sigma'].append(epoch_metrics.get('sigma', 0.0))
            self.history['mu_eq'].append(epoch_metrics.get('mu_eq', 0.0))
            
            # For nonlinear model
            if 'gamma' in epoch_metrics:
                self.history['gamma'].append(epoch_metrics['gamma'])
            if 'beta' in epoch_metrics:
                self.history['beta'].append(epoch_metrics['beta'])
            if 'r0' in epoch_metrics:
                self.history['r0'].append(epoch_metrics['r0'])
            
            # Log progress
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                log_msg = f"Epoch {epoch+1}/{epochs}: loss={epoch_metrics['loss']:.6f}, "
                log_msg += f"data_loss={epoch_metrics['data_loss']:.6f}, "
                log_msg += f"physics_loss={epoch_metrics['physics_loss']:.6f}"
                
                if val_loader is not None:
                    log_msg += f", val_loss={epoch_metrics['val_loss']:.6f}"
                
                # Add physics parameters
                log_msg += f" | alpha={epoch_metrics['alpha']:.6f}, "
                log_msg += f"mu_eq={epoch_metrics['mu_eq']:.2f}"
                
                logger.info(log_msg)
            
            # Early stopping
            if val_loader is not None:
                current_loss = epoch_metrics['val_loss']
            else:
                current_loss = epoch_metrics['loss']
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                # Save best model
                self.save_model(os.path.join(self.models_dir, f"{self.model_name}_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Save final model
        self.save_model(os.path.join(self.models_dir, f"{self.model_name}_final.pt"))
        
        # Save training history
        self.save_history(os.path.join(self.models_dir, f"{self.model_name}_history.json"))
        
        return self.history
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'params': self.model.get_parameters()
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from file."""
        if not os.path.exists(path):
            logger.error(f"Model file {path} does not exist")
            return
        
        # Load model
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Model loaded from {path}")
    
    def save_history(self, path: str) -> None:
        """Save training history to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to serializable format
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                serializable_history[key] = [v.item() for v in value]
            else:
                serializable_history[key] = value
        
        # Save history
        with open(path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {path}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(self.history['epochs'], self.history['loss'], label='Training Loss')
        if any(self.history['val_loss']):
            axes[0, 0].plot(self.history['epochs'], self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss components
        axes[0, 1].plot(self.history['epochs'], self.history['data_loss'], label='Data Loss')
        axes[0, 1].plot(self.history['epochs'], self.history['physics_loss'], label='Physics Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot alpha parameter
        axes[1, 0].plot(self.history['epochs'], self.history['alpha'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('α')
        axes[1, 0].set_title('Mean-Reversion Rate (α)')
        axes[1, 0].grid(True)
        
        # Plot equilibrium rating
        axes[1, 1].plot(self.history['epochs'], self.history['mu_eq'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('μ_eq')
        axes[1, 1].set_title('Equilibrium Rating (μ_eq)')
        axes[1, 1].grid(True)
        
        # Add nonlinear parameters if available
        if self.history['gamma'] and len(self.history['gamma']) > 0:
            fig.set_size_inches(15, 15)
            gs = fig.add_gridspec(3, 2)
            
            # Move existing subplots
            for i in range(2):
                for j in range(2):
                    axes[i, j].set_position(gs[i, j].get_position(fig))
            
            # Add additional row
            ax_gamma = fig.add_subplot(gs[2, 0])
            ax_gamma.plot(self.history['epochs'], self.history['gamma'])
            ax_gamma.set_xlabel('Epoch')
            ax_gamma.set_ylabel('γ')
            ax_gamma.set_title('Nonlinearity (γ)')
            ax_gamma.grid(True)
            
            ax_beta = fig.add_subplot(gs[2, 1])
            ax_beta.plot(self.history['epochs'], self.history['beta'])
            ax_beta.set_xlabel('Epoch')
            ax_beta.set_ylabel('β')
            ax_beta.set_title('Volatility Decay (β)')
            ax_beta.grid(True)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        return fig
    
    def plot_rating_forecast(
        self,
        r0: float,
        t_max: float = 365 * 5,  # 5 years
        n_points: int = 100,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rating forecast.
        
        Args:
            r0: Initial rating
            t_max: Maximum time to forecast (in days)
            n_points: Number of time points
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Generate time points
        t_values = np.linspace(0, t_max, n_points)
        
        # Generate forecast
        forecast = self.model.predict(r0=r0, t_values=t_values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean forecast
        ax.plot(forecast['t'] / 365, forecast['mean'], 'b-', label='Mean Forecast')
        
        # Plot uncertainty bands
        ax.fill_between(
            forecast['t'] / 365, forecast['q05'], forecast['q95'],
            alpha=0.2, color='b', label='90% Confidence Interval'
        )
        
        ax.fill_between(
            forecast['t'] / 365, forecast['q25'], forecast['q75'],
            alpha=0.4, color='b', label='50% Confidence Interval'
        )
        
        # Plot starting point
        ax.scatter([0], [r0], color='r', s=50, zorder=5, label='Starting Rating')
        
        # Add reference line for equilibrium
        params = self.model.get_parameters()
        ax.axhline(params['mu_eq'], color='g', linestyle='--', alpha=0.7, label='Equilibrium Rating')
        
        # Customize plot
        ax.set_xlabel('Years')
        ax.set_ylabel('Rating')
        ax.set_title(f'Rating Forecast from {r0}')
        ax.legend()
        ax.grid(True)
        
        # Add model parameters to text box
        param_text = f"Model Parameters:\n"
        param_text += f"α = {params['alpha']:.6f}\n"
        param_text += f"σ = {params.get('sigma', params.get('sigma0', 0)):.2f}\n"
        param_text += f"μ_eq = {params['mu_eq']:.1f}"
        
        # Add nonlinear parameters if available
        if 'gamma' in params:
            param_text += f"\nγ = {params['gamma']:.2f}"
        if 'beta' in params:
            param_text += f"\nβ = {params['beta']:.6f}"
        
        plt.figtext(0.01, 0.01, param_text, fontsize=9, backgroundcolor='white', alpha=0.8)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Rating forecast plot saved to {save_path}")
        
        return fig
    
    def plot_milestone_predictions(
        self,
        r0: float,
        milestones: List[int],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot milestone predictions.
        
        Args:
            r0: Initial rating
            milestones: List of rating milestones to predict
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Generate time points (20 years)
        t_values = np.linspace(0, 365 * 20, 1000)
        
        # Generate forecast
        forecast = self.model.predict(r0=r0, t_values=t_values)
        
        # Filter milestones that are achievable
        params = self.model.get_parameters()
        mu_eq = params['mu_eq']
        
        # Only include milestones between r0 and mu_eq
        achievable_milestones = [
            m for m in milestones
            if (r0 < m < mu_eq) or (r0 > m > mu_eq)
        ]
        
        # Get times to reach milestones
        milestone_times = {
            milestone: forecast['milestones'].get(milestone, float('inf'))
            for milestone in achievable_milestones
        }
        
        # Filter out unreachable milestones
        milestone_times = {
            k: v for k, v in milestone_times.items()
            if not np.isinf(v) and not np.isnan(v)
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean forecast
        ax.plot(forecast['t'] / 365, forecast['mean'], 'b-', label='Mean Forecast')
        
        # Plot uncertainty bands
        ax.fill_between(
            forecast['t'] / 365, forecast['q05'], forecast['q95'],
            alpha=0.2, color='b', label='90% Confidence Interval'
        )
        
        # Plot starting point
        ax.scatter([0], [r0], color='r', s=50, zorder=5, label='Starting Rating')
        
        # Plot milestones
        for milestone, time in milestone_times.items():
            if time <= t_values[-1]:  # Only plot if milestone is reached within forecast window
                time_years = time / 365
                ax.scatter([time_years], [milestone], color='g', s=80, zorder=5, marker='*')
                ax.axhline(milestone, color='g', linestyle='--', alpha=0.3)
                ax.axvline(time_years, color='g', linestyle='--', alpha=0.3)
                ax.text(time_years + 0.1, milestone + 10, f"{milestone}", fontsize=10)
                ax.text(time_years + 0.1, milestone - 30, f"{time_years:.2f} years", fontsize=8)
        
        # Add reference line for equilibrium
        ax.axhline(mu_eq, color='purple', linestyle='--', alpha=0.7, label='Equilibrium Rating')
        
        # Customize plot
        ax.set_xlabel('Years')
        ax.set_ylabel('Rating')
        ax.set_title(f'Rating Milestones from {r0}')
        ax.legend()
        ax.grid(True)
        
        # Add milestone table
        if milestone_times:
            table_text = "Milestone Predictions:\n"
            for milestone, time in sorted(milestone_times.items()):
                years = time / 365
                table_text += f"{milestone}: {years:.2f} years ({time:.1f} days)\n"
            
            plt.figtext(0.01, 0.01, table_text, fontsize=9, backgroundcolor='white', alpha=0.8)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Milestone prediction plot saved to {save_path}")
        
        return fig
