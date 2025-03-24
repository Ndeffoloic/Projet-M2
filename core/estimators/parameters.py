"""Parameter estimation module for financial models."""
from typing import Tuple, Union

import numpy as np
import pandas as pd


class ParameterEstimator:
    """Parameter estimation for IG-OU and Black-Scholes models."""
    
    @staticmethod
    def estimate_lambda(returns: pd.Series, max_lag: int = 5) -> float:
        """Estimate lambda via autocorrelation decay (eq 3.15 from WCE2009).
        
        Args:
            returns: Series of asset returns
            max_lag: Maximum lag for autocorrelation calculation
            
        Returns:
            float: Estimated lambda (mean reversion rate)
        """
        # Ensure sufficient data
        if len(returns) < max_lag + 10:
            return 0.1  # Default value if insufficient data
            
        # Calculate autocorrelations for different lags
        autocorrs = [returns.autocorr(lag=i) for i in range(1, max_lag+1)]
        
        # Filter out NaN values
        valid_autocorrs = [(i, ac) for i, ac in enumerate(autocorrs, 1) if not np.isnan(ac) and abs(ac) > 1e-10]
        
        if not valid_autocorrs:
            return 0.1  # Default if no valid autocorrelations
            
        # Extract lags and autocorrelation values
        lags, ac_values = zip(*valid_autocorrs)
        lags = np.array(lags).reshape(-1, 1)
        
        # Régression: log(autocorr(k)) ≈ -λ * k
        log_autocorr = np.log(np.abs(ac_values))
        
        # Use least squares to estimate lambda
        try:
            slope, _, _, _ = np.linalg.lstsq(lags, log_autocorr, rcond=None)
            lambda_hat = -slope[0]
        except:
            # Fallback if regression fails
            lambda_hat = 0.1
            
        # Bound lambda to reasonable range
        return max(min(lambda_hat, 1.0), 0.01)
    
    @staticmethod
    def estimate_ig_ab(vol_series: pd.Series, lambda_: float, dt: float = 1.0) -> Tuple[float, float]:
        """Calibrate a and b using exact formula from WCE2009 (eq 3.16)."""
        h = dt
        Y = []
        for k in range(1, len(vol_series)):
            Y_k = vol_series.iloc[k] - np.exp(-lambda_ * h) * vol_series.iloc[k-1]
            Y.append(Y_k)
        
        Y_mean = np.mean(Y)
        Y_var = np.var(Y, ddof=0)  # Variance non biaisée
        
        # Formule exacte de l'article (eq 3.16)
        numerator = Y_var * (np.exp(2 * lambda_ * h) - 1)
        denominator = (np.exp(lambda_ * h) - 1)**3
        
        # Avoid division by zero or negative values
        if denominator <= 1e-10 or numerator <= 0:
            return 0.1, 0.2  # Default values
            
        b_hat = np.sqrt(numerator / denominator)
        a_hat = Y_mean * b_hat / (np.exp(lambda_ * h) - 1)
        
        # Ensure positive values
        a_hat = max(a_hat, 1e-6)
        b_hat = max(b_hat, 1e-6)
        
        return float(a_hat), float(b_hat)

    @staticmethod
    def estimate_igou_parameters(returns: Union[pd.DataFrame, pd.Series]) -> Tuple[float, float, float, float, float]:
        """Estimate parameters for the IG-OU model using method of moments.
        
        Args:
            returns: DataFrame or Series of asset returns. If DataFrame, assumes 'Close' or 'Price' column
            
        Returns:
            tuple: (mu, sigma_sq, lambda_, a, b) parameters for drift, volatility variance, 
                   mean reversion, and IG process parameters
        """
        # Convert DataFrame to Series if needed
        if isinstance(returns, pd.DataFrame):
            # Cherche 'Close' ou 'Price' dans les colonnes
            if 'Close' in returns.columns:
                price_col = 'Close'
            elif 'Price' in returns.columns:
                price_col = 'Price'
            else:
                raise ValueError("DataFrame must contain either a 'Close' or 'Price' column")
            
            returns = returns[price_col].pct_change().dropna()
        elif isinstance(returns, pd.Series):
            returns = returns.pct_change().dropna()
        
        # Ensure sufficient data
        if len(returns) < 30:
            raise ValueError("Insufficient data points for parameter estimation (minimum 30 required)")
        
        # Calculate mean return (drift)
        mu = returns.mean()
        mu = max(min(mu, 0.1), -0.1)  # Bound drift to reasonable range
        
        # Calculate volatility parameters
        sigma_sq = returns.var()  # Variance of returns
        sigma_sq = max(sigma_sq, 1e-6)  # Ensure positive variance
        
        # Estimate lambda using the improved method
        lambda_ = ParameterEstimator.estimate_lambda(returns)
        
        # Estimate a and b using absolute returns as proxy for volatility
        vol_series = returns.abs()
        a_hat, b_hat = ParameterEstimator.estimate_ig_ab(vol_series, lambda_)
        
        return float(mu), float(sigma_sq), float(lambda_), float(a_hat), float(b_hat)
    
    @staticmethod
    def estimate_bs_parameters(returns: Union[pd.DataFrame, pd.Series]) -> Tuple[float, float]:
        """Estimate parameters for the Black-Scholes model.
        
        Args:
            returns: DataFrame or Series of asset returns. If DataFrame, assumes 'Close' or 'Price' column
            
        Returns:
            tuple: (mu, sigma) parameters for drift and volatility
        """
        # Convert DataFrame to Series if needed
        if isinstance(returns, pd.DataFrame):
            # Cherche 'Close' ou 'Price' dans les colonnes
            if 'Close' in returns.columns:
                price_col = 'Close'
            elif 'Price' in returns.columns:
                price_col = 'Price'
            else:
                raise ValueError("DataFrame must contain either a 'Close' or 'Price' column")
            
            returns = returns[price_col].pct_change().dropna()
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
