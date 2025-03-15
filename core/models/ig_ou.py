"""Implementation of the IG-OU model from WCE 2009 paper."""
import numpy as np
from typing import Optional, List

class IGOUModel:
    """Inverse Gaussian Ornstein-Uhlenbeck model implementation.
    
    This implements equation 3.17 from the WCE 2009 paper, with robust parameter validation
    and efficient IG random variable generation.
    """
    
    def __init__(self, lambda_: float, a: float, b: float, mu: Optional[float] = None):
        """Initialize the IG-OU model.
        
        Args:
            lambda_: Mean reversion rate (must be positive)
            a: Shape parameter for IG distribution (must be positive)
            b: Scale parameter for IG distribution (must be positive)
            mu: Optional drift term (defaults to 0)
        """
        self.lambda_ = max(lambda_, 1e-6)  # Ensure positive mean reversion
        self.a = max(a, 1e-10)  # Ensure positive shape
        self.b = max(b, 1e-10)  # Ensure positive scale
        self.mu = mu if mu is not None else 0.0
    
    def generate_ig(self, size: int = 1) -> np.ndarray:
        """Generate Inverse Gaussian random variables.
        
        Implements the algorithm from Section 3.16 of the paper.
        
        Args:
            size: Number of random variables to generate
            
        Returns:
            np.ndarray: Array of IG random variables
        """
        # Generate chi-square variables
        Y = np.random.normal(0, 1, size) ** 2
        
        # Calculate mu = a/b for convenience
        mu = self.a / self.b
        
        # Calculate first candidate
        X1 = mu + (mu**2 * Y)/(2*self.b) - mu/(2*self.b) * np.sqrt(4*mu*self.b*Y + (mu*Y)**2)
        X1 = np.maximum(X1, 1e-10)  # Ensure positivity
        
        # Generate uniform variables for acceptance step
        U = np.random.uniform(0, 1, size)
        
        # Calculate second candidate where needed
        X2 = mu**2/X1
        
        # Select between X1 and X2 based on acceptance criterion
        return np.where(U <= mu/(mu + X1), X1, X2)
    
    def simulate(self, X0: float, T: int, dt: float = 1.0) -> List[float]:
        """Simulate the IG-OU process.
        
        Args:
            X0: Initial volatility value
            T: Number of time steps to simulate
            dt: Time step size (default 1.0 for daily data)
            
        Returns:
            List[float]: Simulated volatility path
        """
        # Initialize path
        path = [max(X0, 1e-10)]  # Ensure positive initial value
        
        # Precompute constants
        exp_lambda = np.exp(-self.lambda_ * dt)
        sqrt_dt = np.sqrt(dt)
        
        # Generate path
        for _ in range(T-1):
            # Current volatility
            X = path[-1]
            
            # Mean and variance for the next step
            mean = self.a/self.b + exp_lambda * (X - self.a/self.b)
            var = X * (1 - exp_lambda**2) / (2 * self.lambda_)
            
            # Generate IG increment
            dW = self.generate_ig()
            
            # Update volatility using the IG-OU SDE
            next_X = mean + np.sqrt(var) * dW
            next_X = max(next_X, 1e-10)  # Ensure positivity
            
            path.append(float(next_X))
        
        return path
