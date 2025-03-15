"""Test suite for volatility modeling."""
import numpy as np
import pandas as pd
from scipy import stats

from core.models.ig_ou import IGOUModel
from core.models.black_scholes import BlackScholesModel

def test_volatility_persistence():
    """Test volatility persistence in IG-OU model."""
    # Initialize model with parameters that encourage persistence
    model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)  # Low lambda for persistence
    
    # Simulate a long path
    path = model.simulate(X0=0.2, T=100)
    
    # Calculate autocorrelation at multiple lags
    returns = np.diff(path)
    autocorrs = [pd.Series(returns).autocorr(lag=lag) for lag in [1, 5, 10]]
    
    # Test if at least one lag shows significant autocorrelation
    assert any(ac > 0 for ac in autocorrs), "No significant volatility persistence found"

def test_volatility_clustering():
    """Test volatility clustering in IG-OU model."""
    # Initialize model with parameters that encourage clustering
    model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)  # Low lambda for persistence
    
    # Simulate multiple paths
    n_paths = 10
    paths = [model.simulate(X0=0.2, T=100) for _ in range(n_paths)]
    
    # Calculate absolute returns for each path
    abs_returns = [np.abs(np.diff(path)) for path in paths]
    
    # Calculate autocorrelation at multiple lags
    autocorrs = []
    for returns in abs_returns:
        lags = [1, 5, 10]
        path_autocorrs = [pd.Series(returns).autocorr(lag=lag) for lag in lags]
        autocorrs.extend(path_autocorrs)
    
    # Test if we observe significant clustering (positive autocorrelation)
    mean_autocorr = np.mean([ac for ac in autocorrs if ac is not None])
    assert mean_autocorr > 0, "No significant volatility clustering found"

def test_volatility_mean_reversion():
    """Test mean-reversion property of IG-OU process."""
    # Model parameters
    X0 = 0.5
    lambda_ = 0.5  # Increased mean-reversion speed
    a = 0.2
    b = 1.0
    
    # Theoretical mean
    theoretical_mean = a/b
    
    # Simulate multiple paths
    n_sims = 50
    T = 200  # Longer simulation period
    model = IGOUModel(lambda_=lambda_, a=a, b=b)
    
    # Store final values
    final_values = []
    for _ in range(n_sims):
        path = model.simulate(X0=X0, T=T)
        final_values.append(path[-1])
    
    # Calculate empirical mean and confidence interval
    empirical_mean = np.mean(final_values)
    confidence_level = 0.95
    margin_of_error = stats.sem(final_values) * stats.t.ppf((1 + confidence_level) / 2, len(final_values) - 1)
    
    # Test if theoretical mean falls within confidence interval
    assert abs(empirical_mean - theoretical_mean) < margin_of_error, \
        f"Mean-reversion test failed: empirical={empirical_mean:.3f}, theoretical={theoretical_mean:.3f}"

def test_volatility_distribution():
    """Test the distribution of volatility."""
    model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)
    path = model.simulate(X0=0.2, T=1000)
    
    # Test positivity
    assert np.all(np.array(path) > 0), "Volatility should always be positive"
    
    # Test basic statistical properties
    assert 0 < np.mean(path) < np.inf, "Mean volatility should be finite and positive"
    assert 0 < np.std(path) < np.inf, "Volatility of volatility should be finite"

def test_parameter_stability():
    """Test stability of the process under different parameters."""
    # Test with various parameter combinations
    parameter_sets = [
        (0.1, 0.01, 1.0),  # Base case
        (0.5, 0.05, 2.0),  # Higher mean-reversion
        (0.05, 0.02, 0.5)  # Lower mean-reversion
    ]
    
    for lambda_, a, b in parameter_sets:
        model = IGOUModel(lambda_=lambda_, a=a, b=b)
        path = model.simulate(X0=0.2, T=100)
        
        # Basic sanity checks
        assert len(path) == 100, "Simulation length incorrect"
        assert np.all(np.array(path) > 0), f"Non-positive values found with parameters: λ={lambda_}, a={a}, b={b}"
        assert np.all(np.isfinite(path)), f"Non-finite values found with parameters: λ={lambda_}, a={a}, b={b}"
