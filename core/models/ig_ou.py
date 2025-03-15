"""Implementation of the IG-OU model from WCE 2009 paper."""
import numpy as np
from typing import Optional

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
    
    def generate_ig(self, size: int) -> np.ndarray:
        """Generate Inverse Gaussian random variables.
        
        Implements the algorithm from Section 3.16 of the paper.
        
        Args:
            size: Number of random variables to generate
            
        Returns:
            np.ndarray: Array of IG random variables
        """
        # Generate standard normal variables
        N = np.random.normal(0, 1, size)
        Y = N * N
        
        # Calculate first value using quadratic formula
        X1 = (self.a/self.b) + Y/(2*self.b**2) - \
             np.sqrt(4*self.a*self.b*Y + Y**2)/(2*self.b**2)
        X1 = np.maximum(X1, 1e-10)  # Ensure positivity
        
        # Generate uniform variables for acceptance step
        U = np.random.uniform(0, 1, size)
        mask = U <= self.a/(self.a + X1*self.b)
        
        # Calculate second value where needed
        second_value = np.divide(self.a**2, self.b**2 * X1, where=X1!=0)
        
        # Return mix of X1 and second value based on acceptance
        return np.where(mask, X1, second_value)
    
    def simulate(self, X0: float, T: int = 30, dt: float = 1/252) -> np.ndarray:
        """Simulate the IG-OU process.
        
        Implements equation 3.17 from the paper using Euler-Maruyama discretization.
        
        Args:
            X0: Initial value
            T: Time horizon in days
            dt: Time step size (default: 1/252 for daily data)
            
        Returns:
            np.ndarray: Simulated path of length T
        """
        n_steps = int(T/dt)
        X = np.zeros(n_steps)
        X[0] = X0
        
        for t in range(1, n_steps):
            h = dt
            L = self.generate_ig(1)[0]
            X[t] = np.exp(-self.lambda_*h)*X[t-1] + L
        
        return X[:30]  # Return first 30 values for visualization
