"""Test suite for financial models."""
import numpy as np
import pandas as pd
import pytest

from core.models.ig_ou import IGOUModel
from core.models.black_scholes import BlackScholesModel
from core.estimators.parameters import ParameterEstimator

def test_igou_model_initialization():
    """Test IG-OU model parameter validation."""
    # Test valid parameters
    model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)
    assert model.lambda_ == 0.1
    assert model.a == 0.01
    assert model.b == 1.0
    
    # Test parameter bounds
    model = IGOUModel(lambda_=-0.1, a=-0.01, b=-1.0)
    assert model.lambda_ >= 1e-6
    assert model.a >= 1e-10
    assert model.b >= 1e-10

def test_igou_simulation():
    """Test IG-OU simulation output."""
    model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)
    path = model.simulate(X0=0.2, T=30)
    
    assert len(path) == 30
    assert np.all(np.isfinite(path))
    assert isinstance(path, np.ndarray)

def test_bs_model_initialization():
    """Test Black-Scholes model parameter validation."""
    # Test valid parameters
    model = BlackScholesModel(mu=0.05, sigma=0.2)
    assert model.mu == 0.05
    assert model.sigma == 0.2
    
    # Test sigma bound
    model = BlackScholesModel(mu=0.05, sigma=-0.2)
    assert model.sigma >= 1e-10

def test_bs_simulation():
    """Test Black-Scholes simulation output."""
    model = BlackScholesModel(mu=0.05, sigma=0.2)
    path = model.simulate(S0=100.0, days=30)
    
    assert len(path) == 30
    assert np.all(np.isfinite(path))
    assert isinstance(path, np.ndarray)

def test_parameter_estimation():
    """Test parameter estimation for both models."""
    # Create sample returns
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
    
    # Test IG-OU parameter estimation
    mu, sigma_sq, lambda_ = ParameterEstimator.estimate_igou_parameters(returns)
    assert isinstance(mu, float)
    assert isinstance(sigma_sq, float)
    assert isinstance(lambda_, float)
    assert lambda_ > 0
    
    # Test Black-Scholes parameter estimation
    mu, sigma = ParameterEstimator.estimate_bs_parameters(returns)
    assert isinstance(mu, float)
    assert isinstance(sigma, float)
    assert sigma > 0

def test_empty_data_handling():
    """Test handling of empty or invalid data."""
    empty_returns = pd.Series([])
    
    # Test IG-OU estimation with empty data
    mu, sigma_sq, lambda_ = ParameterEstimator.estimate_igou_parameters(empty_returns)
    assert mu == 0.0001
    assert sigma_sq == 0.01
    assert lambda_ == 0.1
    
    # Test Black-Scholes estimation with empty data
    mu, sigma = ParameterEstimator.estimate_bs_parameters(empty_returns)
    assert mu == 0.0001
    assert sigma == 0.01
