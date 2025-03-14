import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Helper function to create sample price data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
    prices = [100 * (1 + np.random.normal(0, 0.02)) for _ in range(len(dates))]
    return pd.DataFrame({'Close': prices}, index=dates)

def test_data_format():
    """Test le format des données"""
    df = create_sample_data()
    
    # Vérifier la structure des données
    assert isinstance(df, pd.DataFrame)
    assert 'Close' in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    
    # Vérifier les types de données
    assert df['Close'].dtype.kind in 'iuf'  # integer or float
    assert not df['Close'].isna().any()  # pas de valeurs manquantes

def test_returns_calculation():
    """Test le calcul des rendements"""
    df = create_sample_data()
    returns = df['Close'].pct_change(fill_method=None).dropna()
    
    # Vérifier les propriétés des rendements
    assert len(returns) == len(df) - 1  # un point de moins que les prix
    assert isinstance(returns, pd.Series)
    assert returns.dtype.kind in 'iuf'
    assert not returns.isna().any()

def test_data_consistency():
    """Test la cohérence des données temporelles"""
    df = create_sample_data()
    
    # Vérifier que l'index est trié
    assert df.index.is_monotonic_increasing
    
    # Vérifier qu'il n'y a pas de doublons dans l'index
    assert not df.index.has_duplicates
    
    # Vérifier la fréquence des données
    time_diffs = df.index[1:] - df.index[:-1]
    assert all(td == timedelta(days=1) for td in time_diffs)

def test_price_validation():
    """Test la validation des prix"""
    # Test avec des prix valides
    valid_prices = pd.Series([100.0, 101.0, 99.5])
    assert valid_prices.dtype.kind in 'iuf'
    assert all(valid_prices > 0)
    
    # Test avec des prix négatifs
    negative_prices = pd.Series([100.0, -101.0, 99.5])
    returns = negative_prices.pct_change(fill_method=None).dropna()
    assert len(returns) == len(negative_prices) - 1
