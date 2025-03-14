import numpy as np
import pandas as pd
import pytest
from WCE2009_streamlit import generate_ig, simulate_ig_ou, simulate_bs, estimate_parameters

def test_generate_ig():
    """Test la génération de variables Inverse Gaussiennes"""
    a, b = 1.0, 1.0
    size = 1000
    samples = generate_ig(a, b, size)
    
    # Test basic properties
    assert len(samples) == size
    assert np.all(samples > 0)  # IG distribution is always positive
    assert np.mean(samples) > 0
    
    # Test edge cases
    small_samples = generate_ig(1e-10, 1e-10, 10)
    assert np.all(np.isfinite(small_samples))
    assert np.all(small_samples > 0)

def test_simulate_ig_ou():
    """Test la simulation du processus IG-OU"""
    X0 = 0.1
    lambda_ = 0.5
    a = 1.0
    b = 1.0
    T = 30
    dt = 1/252
    
    trajectory = simulate_ig_ou(X0, lambda_, a, b, T, dt)
    
    # Test basic properties
    assert len(trajectory) == 30  # Should return exactly 30 values
    assert np.all(np.isfinite(trajectory))
    assert np.all(trajectory > 0)  # Volatility should be positive
    
    # Test initial condition
    assert np.isclose(trajectory[0], X0)

def test_simulate_bs():
    """Test la simulation de Black-Scholes"""
    S0 = 100
    mu = 0.05
    sigma = 0.2
    days = 30
    
    prices = simulate_bs(S0, mu, sigma, days)
    
    # Test basic properties
    assert len(prices) == days
    assert prices[0] == S0
    assert np.all(prices > 0)  # Prices should be positive
    assert np.all(np.isfinite(prices))

def test_estimate_parameters():
    """Test l'estimation des paramètres"""
    # Create synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    
    mu, sigma_sq, lambda_ = estimate_parameters(returns)
    
    # Test basic properties
    assert isinstance(mu, float)
    assert isinstance(sigma_sq, float)
    assert isinstance(lambda_, float)
    
    assert np.isfinite(mu)
    assert np.isfinite(sigma_sq)
    assert np.isfinite(lambda_)
    
    assert sigma_sq > 0  # Variance should be positive
    assert lambda_ > 0   # Mean reversion rate should be positive

def test_estimate_parameters_edge_cases():
    """Test les cas limites pour l'estimation des paramètres"""
    # Test empty series
    empty_returns = pd.Series([])
    mu, sigma_sq, lambda_ = estimate_parameters(empty_returns)
    assert mu == 0.0001
    assert sigma_sq == 0.01
    assert lambda_ == 0.1
    
    # Test series with one value
    single_return = pd.Series([0.01])
    mu, sigma_sq, lambda_ = estimate_parameters(single_return)
    assert mu == 0.0001
    assert sigma_sq == 0.01
    assert lambda_ == 0.1
    
    # Test series with NaN values
    nan_returns = pd.Series([0.01, np.nan, 0.02])
    mu, sigma_sq, lambda_ = estimate_parameters(nan_returns)
    assert np.isfinite(mu)
    assert np.isfinite(sigma_sq)
    assert np.isfinite(lambda_)
