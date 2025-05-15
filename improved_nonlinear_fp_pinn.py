"""
Improved Nonlinear Fokker-Planck Physics-Informed Neural Network (PINN) model 
for chess rating dynamics with better stability features.

This module extends the original NonlinearFokkerPlanckNet with additional
features to prevent unrealistic rating collapses:
1. Enforces a minimum rating floor based on starting rating
2. Adds stabilization factor to prevent excessive nonlinearity
3. Improves numerical integration for probability evolution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImprovedNonlinearFokkerPlanckPINN(nn.Module):
    """
    Improved nonlinear Physics-Informed Neural Network for the Fokker-Planck equation
    with rating-dependent drift and diffusion.
    
    This model extends the standard nonlinear model with:
    1. Minimum rating enforcement to prevent unrealistic collapses
    2. Stabilization factor to reduce excessive nonlinearity
    3. Rating-dependent volatility that smoothly decays at higher ratings
    """

    def __init__(self,
                 hidden_layers: List[int] = [64, 128, 128, 64],
                 activation: str = 'tanh',
                 initial_alpha: float = 0.01,
                 initial_gamma: float = 2.0,
                 initial_sigma0: float = 30.0,
                 initial_beta: float = 0.001,
                 initial_mu_eq: float = 2000.0,
                 initial_r0: float = 1500.0,
                 min_rating_fraction: float = 0.9,
                 stabilization_factor: float = 0.5,
                 min_alpha: float = 0.001,
                 max_alpha: float = 0.1,
                 min_gamma: float = 1.1,
                 max_gamma: float = 5.0,
                 min_sigma: float = 5.0,
                 max_sigma: float = 100.0,
                 min_beta: float = 0.0,
                 max_beta: float = 0.01):
        """
        Initialize the improved nonlinear PINN model.

        Args:
            hidden_layers: Sizes of hidden layers in the neural network
            activation: Activation function ('tanh', 'relu', or 'sigmoid')
            initial_alpha: Initial value for drift coefficient
            initial_gamma: Initial value for nonlinearity exponent
            initial_sigma0: Initial value for base volatility
            initial_beta: Initial value for volatility decay rate
            initial_mu_eq: Initial value for equilibrium rating
            initial_r0: Initial value for reference rating
            min_rating_fraction: Minimum allowed rating as fraction of initial rating
            stabilization_factor: Factor to stabilize nonlinearity (0-1, higher = more stable)
            min_alpha/max_alpha: Bounds for alpha parameter
            min_gamma/max_gamma: Bounds for gamma parameter
            min_sigma/max_sigma: Bounds for sigma parameter
            min_beta/max_beta: Bounds for beta parameter
        """
        super().__init__()

        # Parameter bounds
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_beta = min_beta
        self.max_beta = max_beta
        
        # Stability parameters
        self.min_rating_fraction = min_rating_fraction
        self.stabilization_factor = stabilization_factor

        # Initialize unconstrained parameters
        self._alpha_raw = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))
        self._gamma_raw = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))
        self._sigma0_raw = nn.Parameter(torch.tensor(initial_sigma0, dtype=torch.float32))
        self._beta_raw = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        self.mu_eq = nn.Parameter(torch.tensor(initial_mu_eq, dtype=torch.float32))
        self.r0 = nn.Parameter(torch.tensor(initial_r0, dtype=torch.float32))

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
    def alpha(self):
        """Constrained alpha parameter."""
        return torch.sigmoid(self._alpha_raw) * (self.max_alpha - self.min_alpha) + self.min_alpha

    @property
    def gamma(self):
        """Constrained gamma parameter with stabilization."""
        # Apply stabilization factor - brings gamma closer to 1.0 for more stability
        # When stabilization_factor = 0, we get the original gamma
        # When stabilization_factor = 1, we get gamma very close to 1.0 (linear drift)
        raw_gamma = torch.sigmoid(self._gamma_raw) * (self.max_gamma - self.min_gamma) + self.min_gamma
        stabilized_gamma = 1.0 + (raw_gamma - 1.0) * (1.0 - self.stabilization_factor)
        return stabilized_gamma

    @property
    def sigma0(self):
        """Constrained sigma parameter."""
        return torch.sigmoid(self._sigma0_raw) * (self.max_sigma - self.min_sigma) + self.min_sigma
        
    @property
    def sigma(self):
        """Compatibility property for backwards compatibility with the original model."""
        return self.sigma0

    @property
    def beta(self):
        """Constrained beta parameter."""
        return torch.sigmoid(self._beta_raw) * (self.max_beta - self.min_beta) + self.min_beta

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            r: Rating values (batch_size, )
            t: Time values (batch_size, )

        Returns:
            Probability density p(r,t) (batch_size, 1)
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

    def compute_derivatives(self,
                           r: torch.Tensor,
                           t: torch.Tensor,
                           p: torch.Tensor = None) -> dict:
        """Compute derivatives for the Fokker-Planck equation."""
        if p is None:
            p = self.forward(r, t)

        # First-order derivatives
        grad_r = torch.ones_like(p)
        grad_t = torch.ones_like(p)

        # ∂p/∂r
        p_r = torch.autograd.grad(
            p, r, grad_outputs=grad_r, create_graph=True, retain_graph=True
        )[0]

        # ∂p/∂t
        p_t = torch.autograd.grad(
            p, t, grad_outputs=grad_t, create_graph=True, retain_graph=True
        )[0]

        # ∂²p/∂r²
        grad_r2 = torch.ones_like(p_r)
        p_rr = torch.autograd.grad(
            p_r, r, grad_outputs=grad_r2, create_graph=True, retain_graph=True
        )[0]

        return {'p': p, 'p_r': p_r, 'p_t': p_t, 'p_rr': p_rr}

    def drift_term(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute the improved drift term with rating floor to prevent unrealistic drops.
        
        Uses a modified nonlinear drift: -α(r-μ_eq)^γ with a minimum rating floor.
        """
        # Calculate minimum allowed rating (as a fraction of reference rating r0)
        min_rating = self.r0 * self.min_rating_fraction
        
        # Apply soft minimum using sigmoid to blend between actual rating and minimum
        # This creates a smooth transition rather than a hard floor
        blend_factor = torch.sigmoid((r - min_rating) * 0.1)  # Steepness of the blend
        r_effective = blend_factor * r + (1 - blend_factor) * min_rating
        
        # Calculate drift with the effective rating
        r_diff = r_effective - self.mu_eq
        
        # Handle sign for when r < mu_eq (avoid complex numbers)
        sign = torch.sign(r_diff)
        abs_diff = torch.abs(r_diff)
        
        # Compute (r-μ_eq)^γ with proper sign handling
        r_diff_pow = sign * (abs_diff ** self.gamma)
        
        # Final drift term: -α(r-μ_eq)^γ
        return -self.alpha * r_diff_pow

    def volatility_term(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute the improved volatility term.
        
        Returns σ(r)² = σ₀² * e^(-β(r-r₀))
        """
        # Calculate volatility with smoother decay
        sigma_squared = self.sigma0**2 * torch.exp(-self.beta * (r - self.r0))
        
        # Apply a minimum volatility floor to prevent complete stabilization
        min_volatility = self.sigma0 * 0.1  # Minimum of 10% of base volatility
        return torch.max(sigma_squared, torch.tensor(min_volatility**2))

    def fokker_planck_residual(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual of the improved nonlinear Fokker-Planck equation.
        
        The PDE: ∂p/∂t = −∂/∂r[μ(r)p] + (1/2)∂²/∂r²[σ²(r)p]
        """
        # Compute derivatives
        derivs = self.compute_derivatives(r, t)
        p = derivs['p']
        p_r = derivs['p_r']
        p_t = derivs['p_t']
        p_rr = derivs['p_rr']
        
        # Calculate drift and its derivative
        mu = self.drift_term(r)
        
        # Approximate derivative of drift numerically for stability
        eps = 1e-5
        r_plus = r + eps
        mu_plus = self.drift_term(r_plus)
        mu_r = (mu_plus - mu) / eps
        
        # Calculate volatility and its derivative
        sigma_squared = self.volatility_term(r)
        
        # Approximate derivative of volatility numerically
        sigma_squared_plus = self.volatility_term(r_plus)
        sigma_squared_r = (sigma_squared_plus - sigma_squared) / eps
        
        # Drift term in PDE: -∂/∂r[μ(r)p]
        drift_term = -(mu * p_r + p * mu_r)
        
        # Diffusion term in PDE: (1/2)∂²/∂r²[σ²(r)p]
        diffusion_term_1 = sigma_squared * p_rr
        diffusion_term_2 = 2 * sigma_squared_r * p_r
        diffusion_term_3 = p * (sigma_squared_r**2 / sigma_squared)  # Second derivative approximation
        
        diffusion_term = 0.5 * (diffusion_term_1 + diffusion_term_2 + diffusion_term_3)
        
        # Residual of PDE: ∂p/∂t + ∂/∂r[μ(r)p] - (1/2)∂²/∂r²[σ²(r)p]
        residual = p_t + drift_term - diffusion_term
        
        return residual

    def probability_path(self,
                       r0: float,
                       t_values: np.ndarray,
                       r_values: np.ndarray,
                       rk4_steps: int = 10) -> np.ndarray:
        """
        Compute probability density over a grid of r and t values using improved numerical integration.
        
        Args:
            r0: Initial rating
            t_values: Time values to compute probability for
            r_values: Rating values to compute probability for
            rk4_steps: Number of steps for Runge-Kutta integration
            
        Returns:
            2D array of probability density [r_idx, t_idx]
        """
        # Convert to tensors if needed
        if not isinstance(t_values, torch.Tensor):
            t_values = torch.tensor(t_values, dtype=torch.float32)
            
        if not isinstance(r_values, torch.Tensor):
            r_values = torch.tensor(r_values, dtype=torch.float32)
            
        # Get dimensions
        n_r = len(r_values)
        n_t = len(t_values)
        
        # Switch to evaluation mode
        self.eval()
        
        # Initialize arrays to store results
        result = np.zeros((n_r, n_t))
        
        # Create time steps for RK4 integration
        t_max = t_values[-1].item()
        dt = t_max / rk4_steps
        t_steps = torch.linspace(0, t_max, rk4_steps + 1)
        
        # Initial condition: delta function at r0 (approximated by a narrow Gaussian)
        sigma_init = 10.0  # Standard deviation of initial peak
        prob_init = torch.exp(-(r_values - r0)**2 / (2 * sigma_init**2))
        prob_init = prob_init / torch.sum(prob_init)  # Normalize
        
        # Store initial probabilities
        result[:, 0] = prob_init.detach().numpy()
        
        # Runge-Kutta 4th order integration
        for step in range(rk4_steps):
            # Current time
            t_current = t_steps[step]
            
            # Get probabilities at current step
            prob_current = torch.tensor(result[:, step], dtype=torch.float32)
            
            # RK4 integration for one step
            with torch.no_grad():
                # k1 calculation
                k1 = self._calculate_dp_dt(r_values, t_current, prob_current, dt)
                
                # k2 calculation
                prob_k2 = prob_current + 0.5 * dt * k1
                k2 = self._calculate_dp_dt(r_values, t_current + 0.5 * dt, prob_k2, dt)
                
                # k3 calculation
                prob_k3 = prob_current + 0.5 * dt * k2
                k3 = self._calculate_dp_dt(r_values, t_current + 0.5 * dt, prob_k3, dt)
                
                # k4 calculation
                prob_k4 = prob_current + dt * k3
                k4 = self._calculate_dp_dt(r_values, t_current + dt, prob_k4, dt)
                
                # Update probabilities with weighted average
                prob_next = prob_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
                # Ensure probabilities are non-negative and normalized
                prob_next = torch.relu(prob_next)
                if torch.sum(prob_next) > 0:
                    prob_next = prob_next / torch.sum(prob_next)
                    
                # Store result
                next_idx = step + 1
                result[:, next_idx] = prob_next.detach().numpy()
        
        # Interpolate to get values at the desired time points
        # For each r value, interpolate across time points
        interpolated_result = np.zeros((n_r, n_t))
        interpolated_result[:, 0] = result[:, 0]  # Keep initial condition
        
        for i in range(n_r):
            # Linear interpolation in time for each rating value
            for j in range(1, n_t):
                t_target = t_values[j].item()
                # Find closest t_steps values
                idx_before = int(t_target / dt)
                idx_after = min(idx_before + 1, rk4_steps)
                
                # Interpolation factor
                alpha = (t_target - t_steps[idx_before].item()) / dt
                
                # Interpolate
                if idx_before == idx_after:
                    interpolated_result[i, j] = result[i, idx_before]
                else:
                    interpolated_result[i, j] = (1-alpha) * result[i, idx_before] + alpha * result[i, idx_after]
        
        return interpolated_result

    def _calculate_dp_dt(self, r: torch.Tensor, t: torch.Tensor, p: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Calculate dp/dt using the Fokker-Planck equation, for use in numerical integration.
        
        Args:
            r: Rating values
            t: Current time (scalar)
            p: Current probability distribution
            dt: Time step size
            
        Returns:
            dp/dt for each r value
        """
        # Expand t to match r size
        t_expanded = torch.ones_like(r) * t
        
        # Rating step size
        dr = r[1] - r[0]
        
        # Calculate drift and diffusion at each r
        mu = torch.zeros_like(r)
        sigma_squared = torch.zeros_like(r)
        
        for i in range(len(r)):
            mu[i] = self.drift_term(r[i])
            sigma_squared[i] = self.volatility_term(r[i])
        
        # Initialize dp/dt
        dpdt = torch.zeros_like(r)
        
        # Calculate dp/dt using finite difference method
        # Central difference for interior points, forward/backward at boundaries
        for i in range(len(r)):
            # Drift term: -∂/∂r[μ(r)p]
            if i == 0:
                # Forward difference at left boundary
                drift = -(mu[i+1]*p[i+1] - mu[i]*p[i]) / dr
            elif i == len(r) - 1:
                # Backward difference at right boundary
                drift = -(mu[i]*p[i] - mu[i-1]*p[i-1]) / dr
            else:
                # Central difference for interior
                drift = -(mu[i+1]*p[i+1] - mu[i-1]*p[i-1]) / (2*dr)
            
            # Diffusion term: (1/2)∂²/∂r²[σ²(r)p]
            if i == 0:
                # Forward difference for second derivative
                diffusion = 0.5 * (sigma_squared[i+1]*p[i+1] - 2*sigma_squared[i]*p[i] + 0) / (dr**2)
            elif i == len(r) - 1:
                # Backward difference for second derivative
                diffusion = 0.5 * (0 - 2*sigma_squared[i]*p[i] + sigma_squared[i-1]*p[i-1]) / (dr**2)
            else:
                # Central difference for interior
                diffusion = 0.5 * (sigma_squared[i+1]*p[i+1] - 2*sigma_squared[i]*p[i] + sigma_squared[i-1]*p[i-1]) / (dr**2)
            
            # Combine terms
            dpdt[i] = drift + diffusion
        
        return dpdt

    def milestone_probability(self,
                            r0: float,
                            target_rating: float,
                            target_time: float,
                            n_points: int = 1000) -> float:
        """
        Compute the probability of reaching a target rating by a target time.
        
        Args:
            r0: Initial rating
            target_rating: Target rating to reach
            target_time: Time by which to reach the target rating (days)
            n_points: Number of points to use for numerical integration
            
        Returns:
            Probability of reaching the target rating by the target time
        """
        # Range expansion to ensure proper integration
        rating_range = max(500, abs(target_rating - r0) * 2)
        rating_min = min(r0, target_rating) - rating_range / 2
        rating_max = max(r0, target_rating) + rating_range / 2
        
        # Create rating range for integration
        r_values = np.linspace(rating_min, rating_max, n_points)
        
        # Create time values
        t_values = np.array([target_time])
        
        # Compute probability density using our improved method
        p_grid = self.probability_path(r0, t_values, r_values)
        p_values = p_grid[:, 0]  # Extract values at target_time
        
        # Ensure p_values are valid
        if np.any(np.isnan(p_values)):
            logger.warning("NaN values detected in probability calculation. Using fallback method.")
            return 0.5  # Fallback to a neutral probability
        
        # Normalize probability density to ensure it integrates to 1
        total_probability = np.trapz(p_values, r_values)
        if total_probability <= 0 or np.isnan(total_probability):
            logger.warning(f"Invalid total probability: {total_probability}. Using fallback.")
            return 0.5  # Fallback to a neutral probability
        
        normalized_p_values = p_values / total_probability
        
        # Determine integration bounds based on whether target is higher or lower than initial
        if target_rating > r0:
            # For rating increase, compute P(r >= target_rating)
            idx = np.searchsorted(r_values, target_rating)
            if idx >= len(r_values):
                return 0.0  # Target is beyond our integration range
            
            cumulative_prob = np.trapz(normalized_p_values[idx:], r_values[idx:])
        else:
            # For rating decrease or equal, compute P(r <= target_rating)
            idx = np.searchsorted(r_values, target_rating)
            if idx == 0:
                return 0.0  # Target is below our integration range
            
            cumulative_prob = np.trapz(normalized_p_values[:idx], r_values[:idx])
        
        # Ensure we return a valid probability
        return max(0.0, min(1.0, cumulative_prob))

    def robust_milestone_prediction(self,
                                  r0: float,
                                  target_rating: float,
                                  max_time: float = 365.0 * 5) -> float:
        """
        A robust milestone prediction method with improved stability.
        
        Args:
            r0: Starting rating
            target_rating: Target rating to reach
            max_time: Maximum time to consider (days)
            
        Returns:
            Estimated time to reach the milestone (days)
        """
        # If target is very close to or below current rating, return a small time
        if target_rating <= r0 + 5:
            return 30.0  # Return 1 month for essentially already achieved milestones
        
        # Extract model parameters
        alpha = self.alpha.item()
        gamma = self.gamma.item()
        sigma0 = self.sigma0.item()
        beta = self.beta.item()
        mu_eq = self.mu_eq.item()
        
        # If target is way above equilibrium, use a careful approach
        if target_rating > mu_eq + 200:
            # Use historical data velocity-based estimate
            historical_velocity = 0.5  # points per day (reasonable estimate)
            predicted_days = (target_rating - r0) / historical_velocity
            return min(predicted_days, max_time)
        
        try:
            # For the improved nonlinear model, we'll use a time-to-threshold approach
            # Search for when the probability of reaching the milestone exceeds 50%
            
            # Time points to check
            time_points = np.linspace(30, max_time, 20)  # Check 20 time points
            
            # Calculate probabilities at each time point
            probs = []
            for t in time_points:
                prob = self.milestone_probability(r0, target_rating, t)
                probs.append(prob)
            
            # Find first time where probability exceeds threshold
            threshold = 0.5  # 50% probability
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    # Found the threshold crossing
                    t = time_points[i]
                    
                    # Refine estimate with linear interpolation if not the first point
                    if i > 0:
                        t_prev = time_points[i-1]
                        prob_prev = probs[i-1]
                        t_interp = t_prev + (threshold - prob_prev) * (t - t_prev) / (prob - prob_prev)
                        return t_interp
                    
                    return t
            
            # If no threshold crossing found, fall back to analytical approximation
            if r0 < mu_eq:
                # For increasing ratings
                r_diff_0 = abs(r0 - mu_eq)
                r_diff_target = abs(target_rating - mu_eq)
                
                # For the case where γ ≠ 1
                if abs(gamma - 1.0) > 1e-6:  # Not too close to 1
                    # Estimated time based on the nonlinear ODE solution
                    time_factor = (1/(alpha * (gamma-1))) * (1/r_diff_target**(gamma-1) - 1/r_diff_0**(gamma-1))
                    
                    # Adjust for volatility effect - higher volatility makes milestones achievable faster
                    avg_volatility = sigma0 * np.exp(-beta * (target_rating - r0)/2)
                    volatility_adjustment = 1.0 / (1.0 + avg_volatility/200.0)
                    
                    # Apply adjustment (higher volatility reduces expected time)
                    time_estimate = time_factor * volatility_adjustment
                    return max(30.0, min(time_estimate, max_time))
                else:
                    # For γ ≈ 1, fall back to linear approximation
                    time_estimate = (1/alpha) * np.log(r_diff_0 / r_diff_target)
                    return max(30.0, min(time_estimate, max_time))
            else:
                # If current rating is above equilibrium, more realistic prediction
                return max_time / 2  # Return half the maximum time as a reasonable compromise
            
        except Exception as e:
            logger.warning(f"Error in milestone prediction: {e}")
            # Fallback to a simple estimate
            if r0 < mu_eq and target_rating < mu_eq:
                velocity = 0.5  # points per day (reasonable estimate)
                return min((target_rating - r0) / velocity, max_time)
            else:
                return max_time  # Return max time for unrealistic cases
