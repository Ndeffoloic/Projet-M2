import numpy as np

class BSModel:
    """
    Modèle de Black-Scholes pour la simulation des prix
    """
    def __init__(self, mu: float, sigma: float):
        self.mu = self._validate_param(mu, -1.0, 1.0, 'mu')
        self.sigma = self._validate_param(sigma, 1e-10, 2.0, 'sigma')

    def _validate_param(self, value, min_val, max_val, name):
        if not (min_val <= value <= max_val):
            raise ValueError(f"Paramètre {name} hors limites [{min_val}, {max_val}]")
        return value

    def simulate(self, S0: float, T: int = 30, dt: float = 1/252) -> np.ndarray:
        """
        Simule les prix avec un modèle de Black-Scholes
        
        Args:
            S0: Prix initial
            T: Horizon temporel en jours
            dt: Pas de temps (1/252 pour données journalières)
            
        Returns:
            np.ndarray: Trajectoire des prix simulés
        """
        n_steps = int(T/dt)
        prices = np.zeros(n_steps)
        prices[0] = S0
        
        # Génération des incréments browniens
        dW = np.random.normal(0, np.sqrt(dt), n_steps-1)
        
        # Calcul du drift et du terme de diffusion
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dW
        
        # Simulation des prix
        for t in range(1, n_steps):
            prices[t] = prices[t-1] * np.exp(drift + diffusion[t-1])
        
        return prices[:T]  # Garantit exactement T périodes
