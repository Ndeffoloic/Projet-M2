import time
import pytest
import numpy as np
import pandas as pd
from WCE2009_streamlit import generate_ig, simulate_ig_ou, simulate_bs, estimate_parameters

def measure_execution_time(func):
    """Décorateur pour mesurer le temps d'exécution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

@pytest.mark.benchmark
def test_generate_ig_performance():
    """Test la performance de la génération IG"""
    sizes = [100, 1000, 10000]
    times = []
    
    @measure_execution_time
    def run_generate_ig(size):
        return generate_ig(1.0, 1.0, size)
    
    for size in sizes:
        _, execution_time = run_generate_ig(size)
        times.append(execution_time)
        
    # La génération devrait être rapide même pour de grandes tailles
    assert times[-1] < 1.0  # Moins d'une seconde pour 10000 échantillons
    
    # Vérifier la complexité linéaire
    ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
    for ratio in ratios:
        assert ratio < 15  # Le ratio devrait être proche de 10 (ratio des tailles)

@pytest.mark.benchmark
def test_simulate_ig_ou_performance():
    """Test la performance de la simulation IG-OU"""
    @measure_execution_time
    def run_simulation():
        return simulate_ig_ou(0.2, 0.5, 1.0, 1.0, T=30, dt=1/252)
    
    # Exécuter plusieurs simulations pour mesurer la consistance
    n_runs = 5
    times = []
    for _ in range(n_runs):
        _, execution_time = run_simulation()
        times.append(execution_time)
    
    # La simulation devrait être rapide
    assert np.mean(times) < 0.5  # Moyenne inférieure à 0.5 secondes
    assert np.std(times) < 0.1   # Temps d'exécution stable

@pytest.mark.benchmark
def test_multiple_simulations_performance():
    """Test la performance avec plusieurs simulations parallèles"""
    n_sims = [10, 100, 1000]
    times = []
    
    @measure_execution_time
    def run_multiple_sims(n):
        return [simulate_bs(100, 0.05, 0.2, 30) for _ in range(n)]
    
    for n in n_sims:
        _, execution_time = run_multiple_sims(n)
        times.append(execution_time)
    
    # Vérifier la scalabilité
    ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
    for ratio in ratios:
        assert ratio < 15  # Le ratio devrait être proche de 10

@pytest.mark.benchmark
def test_parameter_estimation_performance():
    """Test la performance de l'estimation des paramètres"""
    # Créer des jeux de données de différentes tailles
    sizes = [100, 1000, 10000]
    times = []
    
    @measure_execution_time
    def run_estimation(size):
        returns = pd.Series(np.random.normal(0.001, 0.02, size))
        return estimate_parameters(returns)
    
    for size in sizes:
        _, execution_time = run_estimation(size)
        times.append(execution_time)
    
    # L'estimation devrait être rapide même pour de grandes tailles
    assert times[-1] < 1.0  # Moins d'une seconde pour 10000 points
    
    # Vérifier la complexité
    ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
    for ratio in ratios:
        assert ratio < 15  # Vérifier la scalabilité

@pytest.mark.benchmark
def test_memory_usage():
    """Test l'utilisation de la mémoire"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # En MB
    
    # Exécuter une simulation importante
    _ = simulate_ig_ou(0.2, 0.5, 1.0, 1.0, T=100, dt=1/252)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # En MB
    memory_increase = final_memory - initial_memory
    
    # L'augmentation de mémoire devrait être raisonnable
    assert memory_increase < 100  # Moins de 100MB d'augmentation

def test_numerical_stability():
    """Test la stabilité numérique avec des valeurs extrêmes"""
    # Test avec des valeurs très petites
    small_result = generate_ig(1e-10, 1e-10, 100)
    assert np.all(np.isfinite(small_result))
    
    # Test avec des valeurs très grandes
    large_result = generate_ig(1e10, 1e10, 100)
    assert np.all(np.isfinite(large_result))
    
    # Test de la stabilité de la simulation
    extreme_sim = simulate_ig_ou(1e-10, 1e10, 1e-10, 1e10, T=30, dt=1/252)
    assert np.all(np.isfinite(extreme_sim))
