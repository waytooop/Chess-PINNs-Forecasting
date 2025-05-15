"""
Linear Fokker-Planck Physics-Informed Neural Network for Chess Rating Forecasting.

This module implements the linear Ornstein-Uhlenbeck model with:
- Linear mean-reversion drift term: μ(r) = -α(r-μ_eq)
- Constant volatility: σ²(r) = σ²

The corresponding Fokker-Planck equation is:
∂p/∂t = -∂/∂r[-α(r-μ_eq)p] + (1/2)∂²/∂r²[σ²p]
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LinearFokkerPlanckPINN(nn.Module):
    """
    Linear Fokker-Planck PINN model for rating prediction using an Ornstein-Uhlenbeck process.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 128, 128, 64],
        activation: str = 'tanh',
        initial_alpha: float = 0.01,
        initial_sigma: float = 30.0,
        initial_mu_eq: float = 2800.0,
        clamp_alpha: bool = True,
        clamp_sigma: bool = True
    ):
        """
        Initialize the Linear Fokker-Planck PINN model.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function to use ('tanh', 'relu', or 'sigmoid')
            initial_alpha: Initial value for mean-reversion rate
            initial_sigma: Initial value for volatility
            initial_mu_eq: Initial value for equilibrium rating
            clamp_alpha: Whether to clamp alpha to be positive
            clamp_sigma: Whether to clamp sigma to be positive
        """
        super().__init__()
        
        # Physics parameters
        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha), dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(initial_sigma), dtype=torch.float32))
        self.mu_eq = nn.Parameter(torch.tensor(initial_mu_eq, dtype=torch.float32))
        
        # Whether to clamp parameters
        self.clamp_alpha = clamp_alpha
        self.clamp_sigma = clamp_sigma
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build neural network layers
        layers = []
        
        # Input layer (r, t) -> first hidden layer
        layers.append(nn.Linear(2, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer -> p(r,t)
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.layers = nn.ModuleList(layers)
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get mean-reversion rate."""
        if self.clamp_alpha:
            return torch.exp(self.log_alpha)
        return self.log_alpha
    
    @property
    def sigma(self) -> torch.Tensor:
        """Get volatility."""
        if self.clamp_sigma:
            return torch.exp(self.log_sigma)
        return self.log_sigma
    
    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute probability density.
        
        Args:
            r: Rating tensor
            t: Time tensor
            
        Returns:
            Probability density p(r,t)
        """
        # Combine inputs
        x = torch.stack([r, t], dim=1)
        
        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # Final layer (no activation)
        x = self.layers[-1](x)
        
        # Ensure output is positive (probability density)
        p = F.softplus(x)
        
        return p
    
    def compute_derivatives(
        self, 
        r: torch.Tensor, 
        t: torch.Tensor, 
        p: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute derivatives for the Fokker-Planck equation.
        
        Args:
            r: Rating tensor
            t: Time tensor
            p: Probability density tensor (computed if None)
            
        Returns:
            Dictionary with derivatives: p, p_r, p_t, p_rr
        """
        # Ensure requires_grad
        if not r.requires_grad:
            r = r.detach().clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.detach().clone().requires_grad_(True)
        
        # Forward pass if p not provided
        if p is None:
            p = self.forward(r, t)
        
        # Make sure p is the right shape (N, 1) and create ones with same shape
        p = p.view(-1, 1)
        ones = torch.ones_like(p)
        
        # Compute first-order derivatives
        p_r = torch.autograd.grad(
            p, r, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]
        
        p_t = torch.autograd.grad(
            p, t, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]
        
        # Make sure p_r is the right shape and create ones with same shape for second derivative
        p_r = p_r.view(-1, 1)
        ones_like_p_r = torch.ones_like(p_r)
        
        # Compute second-order derivative (∂²p/∂r²)
        p_rr = torch.autograd.grad(
            p_r, r, grad_outputs=ones_like_p_r, create_graph=True, retain_graph=True
        )[0]
        
        return {
            'p': p,
            'p_r': p_r,
            'p_t': p_t,
            'p_rr': p_rr
        }
    
    def fokker_planck_residual(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual of the Fokker-Planck equation.
        
        Equation:
        ∂p/∂t = -∂/∂r[-α(r-μ_eq)p] + (1/2)∂²/∂r²[σ²p]
        
        Args:
            r: Rating tensor
            t: Time tensor
            
        Returns:
            Residual of the Fokker-Planck equation
        """
        # Compute derivatives
        derivs = self.compute_derivatives(r, t)
        p = derivs['p']
        p_r = derivs['p_r']
        p_t = derivs['p_t']
        p_rr = derivs['p_rr']
        
        # Compute physics terms
        # Drift term: μ(r) = -α(r-μ_eq)
        mu = -self.alpha * (r - self.mu_eq)
        
        # Diffusion term: σ²(r) = σ²
        sigma_squared = self.sigma**2
        
        # Fokker-Planck equation:
        # ∂p/∂t = -∂/∂r[μ(r)p] + (1/2)∂²/∂r²[σ²(r)p]
        
        # ∂/∂r[μ(r)p] = μ(r)∂p/∂r + p∂μ(r)/∂r
        # ∂μ(r)/∂r = -α
        drift_term = mu * p_r + p * (-self.alpha)
        
        # (1/2)∂²/∂r²[σ²(r)p] = (1/2)σ²∂²p/∂r²
        # Assuming σ is constant for the linear model
        diffusion_term = 0.5 * sigma_squared * p_rr
        
        # Residual of PDE
        residual = p_t + drift_term - diffusion_term
        
        return residual
    
    def get_parameters(self) -> Dict[str, float]:
        """Get physical parameters as a dictionary."""
        with torch.no_grad():
            return {
                'alpha': self.alpha.item(),
                'sigma': self.sigma.item(),
                'mu_eq': self.mu_eq.item()
            }
    
    def predict(
        self, 
        r0: float, 
        t_values: np.ndarray, 
        n_samples: int = 1000, 
        device: Optional[torch.device] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict future rating distribution.
        
        Args:
            r0: Initial rating
            t_values: Time values to predict for
            n_samples: Number of points in rating space
            device: Device to use for prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Move model to device if specified
        if device is not None:
            self.to(device)
        
        # Use evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Get parameters
            alpha = self.alpha.item()
            sigma = self.sigma.item()
            mu_eq = self.mu_eq.item()
            
            # For linear Ornstein-Uhlenbeck, we can compute mean and std analytically
            # Mean: μ(t) = μ_eq + (r0 - μ_eq) * exp(-αt)
            means = mu_eq + (r0 - mu_eq) * np.exp(-alpha * t_values)
            
            # Standard deviation: σ(t) = σ * sqrt((1 - exp(-2αt)) / (2α))
            stds = sigma * np.sqrt((1 - np.exp(-2 * alpha * t_values)) / (2 * alpha))
            
            # Create result dictionary
            results = {
                'r0': r0,
                't': t_values,
                'mean': means,
                'std': stds,
                # Add quantiles for uncertainty visualization
                'q05': means - 1.96 * stds,  # 5th percentile
                'q25': means - 0.674 * stds,  # 25th percentile
                'q75': means + 0.674 * stds,  # 75th percentile
                'q95': means + 1.96 * stds,  # 95th percentile
            }
            
            # Compute time to reach specific milestones
            def time_to_reach(target_rating):
                # Only if target is between r0 and mu_eq
                if (r0 < target_rating < mu_eq) or (r0 > target_rating > mu_eq):
                    # Time to reach target: t = -ln((R-μ_eq)/(r0-μ_eq))/α
                    time = -np.log((target_rating - mu_eq) / (r0 - mu_eq)) / alpha
                    return max(0, time)
                elif np.isclose(r0, target_rating):
                    return 0
                else:
                    return float('inf')
            
            # Add milestone predictions
            results['milestones'] = {}
            for milestone in range(2000, 3000, 100):
                results['milestones'][milestone] = time_to_reach(milestone)
            
            return results
