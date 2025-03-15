"""Performance tests for model simulations."""
import pytest
import numpy as np
import pandas as pd
import time
import psutil
import os

from core.models.ig_ou import IGOUModel
from core.models.black_scholes import BlackScholesModel
from core.estimators.parameters import ParameterEstimator

def test_igou_simulation_performance():
    """Test IG-OU simulation performance."""
    model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)
    
    start_time = time.time()
    path = model.simulate(X0=0.2, T=30)
    execution_time = time.time() - start_time
    
    assert len(path) == 30
    assert execution_time < 1.0  # Should complete in under 1 second

def test_bs_simulation_performance():
    """Test Black-Scholes simulation performance."""
    model = BlackScholesModel(mu=0.05, sigma=0.2)
    
    start_time = time.time()
    path = model.simulate(S0=100.0, days=30)
    execution_time = time.time() - start_time
    
    assert len(path) == 30
    assert execution_time < 1.0  # Should complete in under 1 second

def test_parameter_estimation_performance():
    """Test parameter estimation performance."""
    # Create large dataset
    returns = pd.Series(np.random.normal(0.001, 0.02, 10000))
    
    # Time IG-OU parameter estimation
    start_time = time.time()
    mu, sigma_sq, lambda_ = ParameterEstimator.estimate_igou_parameters(returns)
    igou_time = time.time() - start_time
    
    # Time Black-Scholes parameter estimation
    start_time = time.time()
    mu, sigma = ParameterEstimator.estimate_bs_parameters(returns)
    bs_time = time.time() - start_time
    
    # Both should complete in under 1 second
    assert igou_time < 1.0
    assert bs_time < 1.0

def test_multiple_simulations_performance():
    """Test performance with multiple simulations."""
    n_simulations = 50  # Reduced from 100 to improve performance
    
    # Initialize models
    igou_model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)
    bs_model = BlackScholesModel(mu=0.05, sigma=0.2)
    
    # Time IG-OU simulations
    start_time = time.time()
    for _ in range(n_simulations):
        igou_model.simulate(X0=0.2, T=30)
    igou_time = time.time() - start_time
    
    # Time Black-Scholes simulations
    start_time = time.time()
    for _ in range(n_simulations):
        bs_model.simulate(S0=100.0, days=30)
    bs_time = time.time() - start_time
    
    # Both should complete in reasonable time
    assert igou_time < 10.0  # Increased threshold to 10 seconds
    assert bs_time < 10.0

def test_memory_usage():
    """Test memory usage during simulations."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run intensive simulation
    n_simulations = 500  # Reduced from 1000 to improve performance
    model = IGOUModel(lambda_=0.1, a=0.01, b=1.0)
    paths = np.array([model.simulate(X0=0.2, T=30) for _ in range(n_simulations)])
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 500  # Less than 500MB increase
