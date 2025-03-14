import numpy as np
import pandas as pd
from typing import Dict, Union, Optional

class IGParameterEstimator:
    """Estimateur des paramètres selon les équations (3.15) et (3.16) du document"""
    
    @staticmethod
    def estimate(returns: pd.Series, d: int = 5) -> Dict[str, float]:
        """
        Estime les paramètres du modèle IG-OU à partir des rendements
        
        Args:
            returns: Série temporelle des rendements
            d: Nombre de lags pour l'estimation de lambda
            
        Returns:
            dict: Dictionnaire contenant mu, sigma_sq et lambda_
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return {'mu': 0.0001, 'sigma_sq': 0.01, 'lambda_': 0.1}
        
        # Calcul des moments
        mu_hat = returns.mean()
        sigma_sq_hat = 2 * returns.var()
        
        # Calcul de l'autocorrélation
        rho = [returns.autocorr(lag=k) for k in range(1, d+1)]
        rho = [r if not np.isnan(r) else 0.1 for r in rho]
        
        # Estimation lambda selon deux méthodes
        lambda1 = -np.log(max(min(rho[0], 0.999), 1e-6))
        
        # Méthode des moindres carrés pour lambda
        lambda_range = np.linspace(0.01, 1, 100)
        residuals = np.array([
            np.sum((rho - np.exp(-l*np.arange(1,d+1)))**2)
            for l in lambda_range
        ])
        lambda2 = lambda_range[np.argmin(residuals)]
        
        return {
            'mu': mu_hat,
            'sigma_sq': sigma_sq_hat,
            'lambda_': min(lambda1, lambda2)
        }

class BSParameterEstimator:
    """Estimateur des paramètres du modèle Black-Scholes"""
    
    @staticmethod
    def estimate(returns: pd.Series) -> Dict[str, float]:
        """
        Estime mu et sigma à partir des rendements
        
        Args:
            returns: Série temporelle des rendements
            
        Returns:
            dict: Dictionnaire contenant mu et sigma
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return {'mu': 0.0001, 'sigma': 0.01}
        
        # Annualisation des paramètres (252 jours de trading)
        mu = returns.mean() * 252
        sigma = np.sqrt(returns.var() * 252)
        
        return {
            'mu': mu,
            'sigma': sigma
        }
