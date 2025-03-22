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

    def simulate(self, S0: float, T: int = 30, dt: float = 1.0) -> Tuple[List[float], List[float]]:
        """Simulate price and volatility paths.
        
        Args:
            S0: Initial price
            T: Number of time steps to simulate (default: 30)
            dt: Time step size (default: 1.0)
            
        Returns:
            Tuple containing price path and volatility path
        """
        # Generate volatility path
        vol_path = self.igou.simulate(
            X0=self.igou.a/self.igou.b,  # Stationary initial value
            T=T,
            dt=dt
        )
        
        # Limiter les valeurs extrêmes de volatilité (cap à 2.0)
        vol_path = np.clip(vol_path, 0.001, 2.0)
        
        # Initialize price path
        price_path = [float(S0)]
        sqrt_dt = np.sqrt(dt)
        
        for t in range(1, T):
            # Valeurs de sécurité en cas de NaN ou Inf
            if np.isnan(price_path[-1]) or np.isinf(price_path[-1]):
                price_path.append(price_path[-2] if len(price_path) > 1 else S0)
                continue
                
            # Calculate drift component (equation 4.3) avec limite
            vol_squared = min(vol_path[t]**2, 4.0)  # Limiter vol^2
            drift = (self.mu + (self.beta - 0.5) * vol_squared) * dt
            
            # Limiter le drift pour éviter les explosions
            drift = np.clip(drift, -0.2, 0.2)
            
            # Brownian motion increment
            dW = np.random.normal(0, 1)
            
            # Éviter les valeurs extrêmes pour vol_path[t] * sqrt_dt * dW
            vol_shock = np.clip(vol_path[t] * sqrt_dt * dW, -0.5, 0.5)
            
            try:
                # Update price avec protection contre les extrêmes
                next_price = price_path[-1] * np.exp(drift + vol_shock)
                
                # Vérifier si le résultat est valide
                if np.isnan(next_price) or np.isinf(next_price):
                    next_price = price_path[-1] * (1 + self.mu * dt)  # Fallback simple
            except:
                # En cas d'erreur, utiliser une mise à jour plus simple
                next_price = price_path[-1] * (1 + self.mu * dt)
                
            # Ajouter une limite supérieure et inférieure au prix
            next_price = np.clip(next_price, S0 * 0.01, S0 * 100)
            price_path.append(next_price)
            
        return price_path, vol_path