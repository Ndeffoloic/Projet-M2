"""Parameter estimation module for financial models."""
import numpy as np
import pandas as pd

class ParameterEstimator:
    """Parameter estimation for IG-OU and Black-Scholes models."""
    
    @staticmethod
    def estimate_igou_parameters(returns: pd.Series) -> tuple:
        """Estimate parameters for the IG-OU model using method of moments.
        
        Args:
            returns: Series of asset returns
            
        Returns:
            tuple: (mu, sigma_sq, lambda_) parameters
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return 0.0001, 0.01, 0.1
        
        # Calculate mean and variance
        mu = returns.mean()
        sigma_sq = 2 * returns.var()  # Factor of 2 from WCE 2009 paper
        
        # Estimate lambda from autocorrelation
        rho1 = returns.autocorr(lag=1)
        if rho1 is None or np.isnan(rho1):
            rho1 = 0.1
        
        # Ensure rho1 is in valid range for log
        rho1 = max(min(rho1, 0.999), 1e-6)
        lambda_ = -np.log(rho1)
        
        return mu, sigma_sq, lambda_
    
    @staticmethod
    def estimate_bs_parameters(returns: pd.Series) -> tuple:
        """Estimate parameters for the Black-Scholes model.
        
        Args:
            returns: Series of asset returns
            
        Returns:
            tuple: (mu, sigma) parameters
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return 0.0001, 0.01
        
        # Simple mean and standard deviation estimation
        mu = returns.mean()
        sigma = returns.std()
        
        return mu, sigma
