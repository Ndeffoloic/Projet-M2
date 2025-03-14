import numpy as np
from scipy.stats import norm

class IGOUProcess:
    """
    Implémentation rigoureuse du modèle IG-OU selon les spécifications du document WCE 2009
    """
    def __init__(self, lambda_: float, a: float, b: float):
        self.lambda_ = self._validate_param(lambda_, 1e-6, 1e3, 'lambda')
        self.a = self._validate_param(a, 1e-10, 1e3, 'a')
        self.b = self._validate_param(b, 1e-10, 1e3, 'b')

    def _validate_param(self, value, min_val, max_val, name):
        if not (min_val <= value <= max_val):
            raise ValueError(f"Paramètre {name} hors limites [{min_val}, {max_val}]")
        return value

    def generate_ig(self, size: int) -> np.ndarray:
        """Implémentation optimisée de l'algorithme IG du document (page 4)"""
        N = norm.rvs(size=size)
        Y = N**2
        X1 = (self.a/self.b) + Y/(2*self.b**2) - np.sqrt(4*self.a*self.b*Y + Y**2)/(2*self.b**2)
        X1 = np.clip(X1, 1e-10, None)
        
        U = np.random.uniform(0, 1, size)
        mask = U <= self.a/(self.a + X1*self.b)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            X2 = np.square(self.a) / (np.square(self.b) * X1)
        
        return np.where(mask, X1, np.nan_to_num(X2, nan=1e10))

    def simulate(self, X0: float, T: int = 30, dt: float = 1/252) -> np.ndarray:
        """Simulation conforme à l'équation (3.17) du document"""
        n_steps = int(T/dt)
        X = np.zeros(n_steps)
        X[0] = X0
        
        for t in range(1, n_steps):
            h = dt
            L = self.generate_ig(1)[0]
            X[t] = np.exp(-self.lambda_*h)*X[t-1] + L
        
        return X[:int(T)]  # Garantit exactement T périodes
