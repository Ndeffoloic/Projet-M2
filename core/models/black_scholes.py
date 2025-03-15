"""Implementation of the Black-Scholes model for comparison."""
import numpy as np
from typing import List

class BlackScholesModel:
    """Black-Scholes model implementation for price simulation."""
    
    def __init__(self, mu: float, sigma: float):
        """Initialize the Black-Scholes model.
        
        Args:
            mu: Drift parameter (expected return)
            sigma: Volatility parameter
        """
        self.mu = mu
        self.sigma = max(sigma, 1e-10)  # Ensure positive volatility
    
    def simulate(self, S0: float, days: int = 30, dt: float = 1.0) -> List[float]:
        """Simulate price paths using the Black-Scholes model.
        
        Args:
            S0: Initial price
            days: Number of days to simulate
            dt: Time step size (default 1.0 for daily data)
            
        Returns:
            List[float]: Simulated price path
        """
        # Initialize path
        path = [float(S0)]
        
        # Precompute constants
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        vol_sqrt_dt = self.sigma * np.sqrt(dt)
        
        # Generate path
        for _ in range(days-1):
            # Generate random shock
            Z = np.random.normal(0, 1)
            
            # Update price using log-normal process
            next_price = path[-1] * np.exp(drift + vol_sqrt_dt * Z)
            path.append(float(next_price))
        
        return path
