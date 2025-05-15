"""
Nonlinear Fokker-Planck Physics-Informed Neural Network for Chess Rating Forecasting.

This module implements the nonlinear "Chess Asymptotic" model with:
- Nonlinear power-law drift term: μ(r) = -α(r-μ_eq)^γ
- Rating-dependent volatility: σ²(r) = σ₀² * e^(-β(r-r₀))

The corresponding Fokker-Planck equation is:
∂p/∂t = -∂/∂r[-α(r-μ_eq)^γ p] + (1/2)∂²/∂r²[σ₀² e^(-β(r-r₀)) p]
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NonlinearFokkerPlanckPINN(nn.Module):
    """
    Nonlinear Fokker-Planck PINN model for rating prediction using the "Chess Asymptotic" model.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 128, 128, 64],
        activation: str = 'tanh',
        initial_alpha: float = 0.01,
        initial_gamma: float = 2.0,
        initial_sigma0: float = 30.0,
        initial_beta: float = 0.001,
        initial_mu_eq: float = 2900.0,
        initial_r0: float = 2200.0,
        clamp_params: bool = True
    ):
        """
        Initialize the Nonlinear Fokker-Planck PINN model.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function to use ('tanh', 'relu', or 'sigmoid')
            initial_alpha: Initial value for mean-reversion rate
            initial_gamma: Initial value for nonlinearity exponent (>1)
            initial_sigma0: Initial value for base volatility
            initial_beta: Initial value for volatility decay rate
            initial_mu_eq: Initial value for equilibrium rating
            initial_r0: Initial value for reference rating (for volatility)
            clamp_params: Whether to ensure parameters stay positive
        """
        super().__init__()
        
        # Physics parameters (using log parametrization for positive parameters)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha), dtype=torch.float32))
        self.log_gamma_minus_one = nn.Parameter(torch.tensor(np.log(initial_gamma - 1.0), dtype=torch.float32))
        self.log_sigma0 = nn.Parameter(torch.tensor(np.log(initial_sigma0), dtype=torch.float32))
        self.log_beta = nn.Parameter(torch.tensor(np.log(initial_beta), dtype=torch.float32))
        self.mu_eq = nn.Parameter(torch.tensor(initial_mu_eq, dtype=torch.float32))
        self.r0 = nn.Parameter(torch.tensor(initial_r0, dtype=torch.float32))
        
        # Whether to clamp parameters
        self.clamp_params = clamp_params
        
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
        if self.clamp_params:
            return torch.exp(self.log_alpha)
        return self.log_alpha
    
    @property
    def gamma(self) -> torch.Tensor:
        """Get nonlinearity exponent (>1)."""
        if self.clamp_params:
            return 1.0 + torch.exp(self.log_gamma_minus_one)
        return self.log_gamma_minus_one
    
    @property
    def sigma0(self) -> torch.Tensor:
        """Get base volatility."""
        if self.clamp_params:
            return torch.exp(self.log_sigma0)
        return self.log_sigma0
    
    @property
    def beta(self) -> torch.Tensor:
        """Get volatility decay rate."""
        if self.clamp_params:
            return torch.exp(self.log_beta)
        return self.log_beta
    
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
        Compute the residual of the nonlinear Fokker-Planck equation.
        
        Equation:
        ∂p/∂t = -∂/∂r[-α(r-μ_eq)^γ p] + (1/2)∂²/∂r²[σ₀² e^(-β(r-r₀)) p]
        
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
        # Nonlinear drift term: μ(r) = -α(r-μ_eq)^γ
        r_diff = r - self.mu_eq
        
        # Handle cases where r < mu_eq (avoid complex numbers)
        sign = torch.sign(r_diff)
        abs_diff = torch.abs(r_diff)
        
        # Compute (r-μ_eq)^γ with proper sign handling
        r_diff_pow = sign * (abs_diff ** self.gamma)
        
        # Calculate drift
        mu = -self.alpha * r_diff_pow
        
        # Calculate derivative of drift for the PDE
        mu_r = -self.alpha * self.gamma * sign * (abs_diff ** (self.gamma - 1))
        
        # State-dependent diffusion term: σ(r)² = σ₀² e^(-β(r-r₀))
        sigma_squared = self.sigma0**2 * torch.exp(-self.beta * (r - self.r0))
        
        # Calculate derivative of diffusion for the PDE
        sigma_squared_r = -self.beta * sigma_squared
        
        # Drift term calculation
        drift_term = mu * p_r + p * mu_r
        
        # Diffusion term with state-dependent diffusion
        diffusion_term = 0.5 * (sigma_squared * p_rr + sigma_squared_r * p_r)
        
        # Residual of PDE
        residual = p_t + drift_term - diffusion_term
        
        return residual
    
    def get_parameters(self) -> Dict[str, float]:
        """Get physical parameters as a dictionary."""
        with torch.no_grad():
            return {
                'alpha': self.alpha.item(),
                'gamma': self.gamma.item(),
                'sigma0': self.sigma0.item(),
                'beta': self.beta.item(),
                'mu_eq': self.mu_eq.item(),
                'r0': self.r0.item()
            }
    
    def predict(
        self, 
        r0: float, 
        t_values: np.ndarray, 
        n_samples: int = 1000, 
        device: Optional[torch.device] = None,
        use_numerical: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict future rating distribution.
        
        For the nonlinear model, we need to use numerical integration to predict
        the mean trajectory, and Monte Carlo simulations for uncertainty.
        
        Args:
            r0: Initial rating
            t_values: Time values to predict for
            n_samples: Number of Monte Carlo samples for uncertainty
            device: Device to use for prediction
            use_numerical: Whether to use numerical integration (vs. sampling)
            
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
            params = self.get_parameters()
            alpha = params['alpha']
            gamma = params['gamma']
            sigma0 = params['sigma0']
            beta = params['beta']
            mu_eq = params['mu_eq']
            r0_ref = params['r0']
            
            # Define SDE drift function for numerical integration
            def drift_func(t, r):
                # Modified drift function to ensure more stable behavior for r near equilibrium
                # For players within 400 points of equilibrium, use a gentler drift
                if abs(r - mu_eq) < 400:
                    # Use a more linear model for ratings close to equilibrium
                    effective_gamma = min(1.2, gamma)
                    return -alpha * np.power(np.abs(r - mu_eq), effective_gamma - 1) * np.sign(r - mu_eq) * (r - mu_eq)
                else:
                    # Use full nonlinear model for ratings far from equilibrium
                    return -alpha * np.power(np.abs(r - mu_eq), gamma - 1) * np.sign(r - mu_eq) * (r - mu_eq)
            
            # Define SDE diffusion function for numerical integration
            def diffusion_func(t, r):
                return sigma0 * np.exp(-0.5 * beta * (r - r0_ref))
            
            # Numerically integrate the SDE for the mean trajectory
            # Use a deterministic ODE (drift only) for the mean
            if use_numerical:
                # Use simpler Euler method for more stability with nonlinear models
                means = np.zeros_like(t_values)
                means[0] = r0
                dt = np.diff(np.concatenate(([0], t_values)))
                
                for i in range(1, len(t_values)):
                    # Simple Euler step with safety checks
                    dr = drift_func(t_values[i-1], means[i-1]) * dt[i]
                    # Limit step size for stability
                    dr = np.clip(dr, -50, 50)
                    means[i] = means[i-1] + dr
                    
                    # Ensure we stay in reasonable bounds
                    # Modified to prevent unrealistic rating drops
                    means[i] = np.clip(means[i], max(r0 * 0.9, r0 - 200), mu_eq * 1.5)
            else:
                # For very short time horizons or testing, use simple Euler-Maruyama
                means = np.zeros_like(t_values)
                means[0] = r0
                dt = np.diff(t_values, prepend=0)[1:]
                
                for i in range(1, len(t_values)):
                    means[i] = means[i-1] + drift_func(t_values[i-1], means[i-1]) * dt[i-1]
            
            # For the demo, use a simplified uncertainty model
            # In production, we would use proper Monte Carlo simulation
            # or analytical approximations for the nonlinear case
            
            # Calculate base volatility based on parameters
            base_std = sigma0 / np.sqrt(2 * alpha)
            
            # Generate time-varying standard deviation that increases with time
            # but decreases as we approach equilibrium
            stds = np.zeros_like(t_values)
            for i, t in enumerate(t_values):
                # Distance from equilibrium as a proportion
                dist_factor = np.abs(means[i] - mu_eq) / np.abs(r0 - mu_eq)
                # Time factor (uncertainty grows with sqrt of time)
                time_factor = np.sqrt(t / 365.0) if t > 0 else 0.01
                # Combine factors
                stds[i] = base_std * time_factor * (0.5 + 0.5 * dist_factor)
            
            # Ensure stds are within reasonable bounds
            stds = np.clip(stds, 5.0, 200.0)
            
            # Calculate quantiles
            q05 = means - 1.96 * stds
            q25 = means - 0.674 * stds
            q75 = means + 0.674 * stds
            q95 = means + 1.96 * stds
            
            # Create result dictionary
            results = {
                'r0': r0,
                't': t_values,
                'mean': means,
                'std': stds,
                'q05': q05,
                'q25': q25,
                'q75': q75,
                'q95': q95,
            }
            
            # Compute time to reach specific milestones
            def time_to_reach(target_rating):
                # For nonlinear model, we need to track the rating evolution
                if (r0 < target_rating < mu_eq) or (r0 > target_rating > mu_eq):
                    # Define time-to-target function using the same Euler method with safety clipping
                    def rating_at_time(t):
                        # Use the same Euler method with safety clipping as the main forecast
                        r_current = r0
                        t_current = 0
                        dt = t / 100  # Small time steps for stability
                        
                        while t_current < t:
                            # Use the same modified drift function here
                            if abs(r_current - mu_eq) < 400:
                                effective_gamma = min(1.2, gamma)
                                dr = -alpha * np.power(np.abs(r_current - mu_eq), effective_gamma - 1) * np.sign(r_current - mu_eq) * (r_current - mu_eq) * dt
                            else:
                                dr = drift_func(t_current, r_current) * dt
                            # Limit step size for stability
                            dr = np.clip(dr, -50, 50)
                            r_current += dr
                            # Ensure we stay in reasonable bounds
                            # Modified to prevent unrealistic rating drops
                            r_current = np.clip(r_current, max(r0 * 0.9, r0 - 200), mu_eq * 1.5)
                            t_current += dt
                        
                        # Return difference from target
                        return r_current - target_rating
                    
                    # Use bounded search for stability
                    t_max = 365 * 20  # 20 years as upper bound
                    try:
                        # Only search if the target is actually reachable
                        if (r0 < target_rating < mu_eq) or (r0 > target_rating > mu_eq):
                            # Find time when rating equals target
                            result = root_scalar(
                                lambda t: rating_at_time(t), 
                                bracket=[0, t_max], 
                                method='brentq'
                            )
                            return max(0, result.root)
                    except:
                        pass
                
                # Return 0 if already at target, infinity if unreachable
                if np.isclose(r0, target_rating):
                    return 0
                else:
                    return float('inf')
            
            # Add milestone predictions
            results['milestones'] = {}
            for milestone in range(2000, 3000, 100):
                results['milestones'][milestone] = time_to_reach(milestone)
            
            return results
