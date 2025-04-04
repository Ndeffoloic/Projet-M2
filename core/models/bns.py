"""Implementation of Barndorff-Nielsen & Shephard model."""
from typing import List, Tuple

import numpy as np
from core.models.ig_ou import IGOUModel


class BNSModel:
    """BNS model combining IG-OU volatility with price dynamics."""
    
    def __init__(self, lambda_: float, a: float, b: float, mu: float, rho: float = -0.5):
        """Initialize the BNS model with parameters aligned with WCE2009 paper.
        
        Args:
            lambda_: Mean reversion rate (must be positive)
            a: Shape parameter for IG distribution (must be positive)
            b: Scale parameter for IG distribution (must be positive)
            mu: Drift parameter
            rho: Leverage effect parameter (default: -0.5, typically negative)
        """
        self.lambda_ = max(lambda_, 1e-6)
        self.a = max(a, 1e-10)
        self.b = max(b, 1e-10)
        self.mu = mu
        self.rho = rho  # Leverage effect parameter (typically negative)
        
        # Initialize IG-OU model for volatility process
        self.igou = IGOUModel(lambda_, a, b)
        
    def generate_levy_jumps(self, T: int, dt: float = 1.0) -> np.ndarray:
        """Generate Lévy jumps for the volatility process.
        
        This implements the compound Poisson process with IG distributed jumps
        as described in the BNS model.
        
        Args:
            T: Number of time steps
            dt: Time step size
            
        Returns:
            np.ndarray: Array of jump sizes
        """
        # Intensity of the Poisson process (average number of jumps per unit time)
        intensity = self.a * dt
        
        # Generate number of jumps for each time step (Poisson distributed)
        num_jumps = np.random.poisson(intensity, T)
        
        # Initialize jumps array
        jumps = np.zeros(T)
        
        # Generate jump sizes for each time step
        for t in range(T):
            if num_jumps[t] > 0:
                # Generate IG distributed jump sizes
                jump_sizes = self.igou.generate_ig(num_jumps[t])
                jumps[t] = np.sum(jump_sizes)
        
        return jumps

    def simulate(self, S0: float, T: int = 30, dt: float = 1.0) -> Tuple[List[float], List[float]]:
        """Simulate price and volatility paths with proper Lévy jumps.
        
        Args:
            S0: Initial price
            T: Number of time steps to simulate (default: 30)
            dt: Time step size (default: 1.0)
            
        Returns:
            Tuple containing price path and volatility path
        """
        # Generate volatility path with IG-OU process
        vol_path = self.igou.simulate(
            X0=self.a/self.b,  # Stationary initial value
            T=T,
            dt=dt
        )
        
        # Generate Lévy jumps for additional fat tails
        levy_jumps = self.generate_levy_jumps(T, dt)
        
        # Add jumps to volatility path (ensuring positivity)
        for t in range(T):
            vol_path[t] = max(vol_path[t] + levy_jumps[t], 1e-6)
        
        # Initialize price path
        price_path = [float(S0)]
        sqrt_dt = np.sqrt(dt)
        
        # Generate correlated Brownian motions for price and volatility
        dW1 = np.random.normal(0, 1, T)  # Price Brownian motion
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, T)  # Volatility Brownian motion
        
        for t in range(1, T):
            # Current price and volatility
            S_prev = price_path[-1]
            sigma_t = np.sqrt(vol_path[t])
            
            # Calculate drift component with risk premium
            drift = (self.mu - 0.5 * vol_path[t]) * dt
            
            # Calculate diffusion component with leverage effect
            diffusion = sigma_t * sqrt_dt * dW1[t-1]
            
            # Calculate log-return
            log_return = drift + diffusion
            
            # Update price using log-normal dynamics
            next_price = S_prev * np.exp(log_return)
            
            # Ensure price is valid
            if np.isnan(next_price) or np.isinf(next_price):
                next_price = S_prev  # Maintain previous price if invalid
            
            price_path.append(float(next_price))
        
        return price_path, vol_path