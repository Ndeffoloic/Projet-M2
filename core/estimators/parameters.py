"""Parameter estimation module for financial models."""
import numpy as np
import pandas as pd
from typing import Tuple, Union

class ParameterEstimator:
    """Parameter estimation for IG-OU and Black-Scholes models."""
    
    @staticmethod
    def estimate_igou_parameters(returns: Union[pd.DataFrame, pd.Series]) -> Tuple[float, float, float]:
        """Estimate parameters for the IG-OU model using method of moments.
        
        Args:
            returns: DataFrame or Series of asset returns. If DataFrame, assumes 'Close' column
            
        Returns:
            tuple: (mu, sigma_sq, lambda_) parameters for drift, volatility variance, and mean reversion
        """
        # Convert DataFrame to Series if needed
        if isinstance(returns, pd.DataFrame):
            if 'Close' not in returns.columns:
                raise ValueError("DataFrame must contain a 'Close' column")
            returns = returns['Close'].pct_change().dropna()
        elif isinstance(returns, pd.Series):
            returns = returns.pct_change().dropna()
        
        # Ensure sufficient data
        if len(returns) < 30:
            raise ValueError("Insufficient data points for parameter estimation (minimum 30 required)")
        
        # Calculate mean return (drift)
        mu = returns.mean()
        mu = max(min(mu, 0.1), -0.1)  # Bound drift to reasonable range
        
        # Calculate volatility parameters
        abs_returns = np.abs(returns)
        sigma_sq = abs_returns.var()  # Volatility of volatility
        sigma_sq = max(sigma_sq, 1e-6)  # Ensure positive variance
        
        # Estimate lambda from volatility persistence
        vol_series = pd.Series(abs_returns)
        rho1 = vol_series.autocorr(lag=1)
        if rho1 is None or np.isnan(rho1) or rho1 <= 0:
            lambda_ = 0.1  # Default if no clear mean reversion
        else:
            # Transform correlation to mean reversion rate
            rho1 = min(rho1, 0.99)  # Ensure finite lambda
            lambda_ = -np.log(rho1)
            lambda_ = max(min(lambda_, 1.0), 0.01)  # Bound lambda to reasonable range
        
        return float(mu), float(sigma_sq), float(lambda_)
    
    @staticmethod
    def estimate_bs_parameters(returns: Union[pd.DataFrame, pd.Series]) -> Tuple[float, float]:
        """Estimate parameters for the Black-Scholes model.
        
        Args:
            returns: DataFrame or Series of asset returns. If DataFrame, assumes 'Close' column
            
        Returns:
            tuple: (mu, sigma) parameters for drift and volatility
        """
        # Convert DataFrame to Series if needed
        if isinstance(returns, pd.DataFrame):
            if 'Close' not in returns.columns:
                raise ValueError("DataFrame must contain a 'Close' column")
            returns = returns['Close'].pct_change().dropna()
        elif isinstance(returns, pd.Series):
            returns = returns.pct_change().dropna()
        
        # Ensure sufficient data
        if len(returns) < 30:
            raise ValueError("Insufficient data points for parameter estimation (minimum 30 required)")
        
        # Calculate annualized parameters
        trading_days = 252
        mu = returns.mean() * trading_days
        sigma = returns.std() * np.sqrt(trading_days)
        
        # Bound parameters to reasonable ranges
        mu = max(min(mu, 0.5), -0.5)  # Limit drift to ±50% annually
        sigma = max(min(sigma, 1.0), 0.01)  # Limit volatility to 1-100% annually
        
        return float(mu), float(sigma)
