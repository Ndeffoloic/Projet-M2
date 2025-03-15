"""Implementation of the Black-Scholes model for comparison."""
import numpy as np

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
    
    def simulate(self, S0: float, days: int = 30) -> np.ndarray:
        """Simulate price paths using the Black-Scholes model.
        
        Args:
            S0: Initial price
            days: Number of days to simulate
            
        Returns:
            np.ndarray: Array of simulated prices
        """
        dt = 1/252  # Daily time step
        prices = [S0]
        
        for _ in range(days-1):
            # Calculate drift and diffusion terms
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            shock = self.sigma * np.sqrt(dt) * np.random.normal()
            
            # Update price using log-normal process
            prices.append(prices[-1] * np.exp(drift + shock))
        
        return np.array(prices)
