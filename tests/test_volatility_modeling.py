import numpy as np
import pandas as pd
import pytest
from scipy import stats
from WCE2009_streamlit import generate_ig, simulate_ig_ou, estimate_parameters

def test_volatility_persistence():
    """Test la persistance de la volatilité dans le modèle IG-OU"""
    X0 = 0.2
    lambda_ = 0.1  # Faible mean-reversion pour tester la persistance
    a = 1.0
    b = 1.0
    T = 30
    dt = 1/252
    
    # Simuler plusieurs trajectoires
    n_sims = 100
    trajectories = np.array([simulate_ig_ou(X0, lambda_, a, b, T, dt) for _ in range(n_sims)])
    
    # Tester l'autocorrélation
    mean_autocorr = np.mean([pd.Series(traj).autocorr(lag=1) for traj in trajectories])
    assert mean_autocorr > 0  # Devrait montrer une autocorrélation positive

def test_volatility_clustering():
    """Test le clustering de volatilité"""
    X0 = 0.2
    lambda_ = 0.5
    a = 1.0
    b = 1.0
    
    # Simuler une longue trajectoire
    trajectory = simulate_ig_ou(X0, lambda_, a, b, T=100, dt=1/252)
    
    # Calculer les différences absolues
    abs_diff = np.abs(np.diff(trajectory))
    
    # Tester l'autocorrélation des différences absolues
    autocorr = pd.Series(abs_diff).autocorr(lag=1)
    assert autocorr > 0  # Les périodes de haute/basse volatilité devraient se regrouper

def test_volatility_mean_reversion():
    """Test la propriété de mean-reversion du processus IG-OU"""
    X0 = 0.5  # Valeur initiale élevée
    lambda_ = 1.0  # Fort mean-reversion
    a = 0.2
    b = 1.0
    
    # Simuler plusieurs trajectoires
    n_sims = 100
    trajectories = np.array([simulate_ig_ou(X0, lambda_, a, b, T=50, dt=1/252) for _ in range(n_sims)])
    
    # Calculer la moyenne théorique (a/b)
    theoretical_mean = a/b
    
    # Vérifier que la moyenne empirique converge vers la moyenne théorique
    final_means = trajectories[:, -1].mean()
    assert np.abs(final_means - theoretical_mean) < 0.1

def test_volatility_distribution():
    """Test les propriétés de la distribution de la volatilité"""
    # Générer un large échantillon
    n_samples = 10000
    a, b = 1.0, 1.0
    samples = generate_ig(a, b, n_samples)
    
    # Tester les propriétés statistiques
    mean = np.mean(samples)
    variance = np.var(samples)
    skewness = stats.skew(samples)
    
    # Vérifier les moments théoriques de la distribution IG
    assert np.abs(mean - a/b) < 0.1
    assert skewness > 0  # La distribution IG est asymétrique à droite

def test_parameter_stability():
    """Test la stabilité des estimations de paramètres"""
    # Créer des données synthétiques
    np.random.seed(42)
    n_samples = 1000
    true_mu = 0.001
    true_sigma = 0.02
    returns = pd.Series(np.random.normal(true_mu, true_sigma, n_samples))
    
    # Estimer les paramètres sur différentes fenêtres
    window_sizes = [100, 200, 500]
    estimates = []
    for window in window_sizes:
        mu, sigma_sq, lambda_ = estimate_parameters(returns.iloc[-window:])
        estimates.append((mu, sigma_sq, lambda_))
    
    # Vérifier la stabilité des estimations
    mu_estimates = [e[0] for e in estimates]
    sigma_estimates = [np.sqrt(e[1]/2) for e in estimates]  # Convert back to sigma
    
    # Les estimations devraient être relativement stables
    assert np.std(mu_estimates) < 0.001
    assert np.std(sigma_estimates) < 0.01
