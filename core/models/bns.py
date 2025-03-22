"""Implementation of Barndorff-Nielsen & Shephard model."""
from typing import List, Tuple

import numpy as np
from core.models.ig_ou import IGOUModel


class BNSModel:
    """BNS model combining IG-OU volatility with price dynamics."""
    
    def __init__(self, igou_model: IGOUModel, mu: float, beta: float):
        """
        Args:
            igou_model: Initialized IG-OU model for volatility
            mu: Drift parameter from paper's equation 4.3
            beta: Skewness parameter
        """
        self.igou = igou_model
        self.mu = mu
        self.beta = beta

    def simulate(self, S0: float, days: int = 30, dt: float = 1.0) -> Tuple[List[float], List[float]]:
        """Simulate price and volatility paths."""
        # Generate volatility path
        vol_path = self.igou.simulate(
            X0=self.igou.a/self.igou.b,  # Stationary initial value
            T=days,
            dt=dt
        )
        
        # Initialize price path
        price_path = [float(S0)]
        sqrt_dt = np.sqrt(dt)
        
        for t in range(1, days):
            # Calculate drift component (equation 4.3)
            drift = (self.mu + (self.beta - 0.5) * vol_path[t]**2) * dt
            
            # Brownian motion increment
            dW = np.random.normal(0, 1)
            
            # Update price
            next_price = price_path[-1] * np.exp(drift + vol_path[t] * sqrt_dt * dW)
            price_path.append(next_price)
            
        return price_path, vol_path