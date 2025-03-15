import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Union

def display_error(message: str) -> None:
    """Affiche un message d'erreur"""
    st.error(f" {message}")

def display_success(message: str) -> None:
    """Affiche un message de succès"""
    st.success(f" {message}")

def display_info(message: str) -> None:
    """Affiche un message d'information"""
    st.info(f" {message}")

def display_parameters(mu: float, sigma_sq: float, lambda_: float):
    """Display estimated model parameters.
    
    Args:
        mu: Drift parameter
        sigma_sq: Volatility variance
        lambda_: Mean reversion speed
    """
    st.subheader("Model Parameters")
    
    params = {
        'μ (Drift)': mu,
        'σ² (Volatility Variance)': sigma_sq,
        'λ (Mean Reversion Speed)': lambda_
    }
    
    st.dataframe(pd.DataFrame({
        'Parameter': list(params.keys()),
        'Value': [f"{v:.6f}" for v in params.values()]
    }).set_index('Parameter'))
    
    with st.expander("Parameter Interpretation"):
        st.markdown("""
        - **μ (Drift)**: Average trend in the asset price
        - **σ² (Volatility Variance)**: Measure of volatility fluctuation
        - **λ (Mean Reversion Speed)**: Rate at which volatility returns to its mean
        """)

def format_data_summary(data: pd.Series) -> None:
    """
    Affiche un résumé des données
    
    Args:
        data: Série temporelle à analyser
    """
    st.subheader("Résumé des données")
    
    summary = pd.DataFrame({
        'Statistique': [
            'Nombre d\'observations',
            'Première date',
            'Dernière date',
            'Prix minimum',
            'Prix maximum',
            'Prix moyen',
            'Écart-type',
            'Skewness',
            'Kurtosis'
        ],
        'Valeur': [
            len(data),
            data.index.min().strftime('%Y-%m-%d'),
            data.index.max().strftime('%Y-%m-%d'),
            f"{data.min():.2f}",
            f"{data.max():.2f}",
            f"{data.mean():.2f}",
            f"{data.std():.2f}",
            f"{data.skew():.2f}",
            f"{data.kurtosis():.2f}"
        ]
    })
    
    st.dataframe(summary, hide_index=True)

def check_data_quality(data: pd.Series) -> tuple[bool, list[str]]:
    """Check data quality.
    
    Args:
        data: Time series data to check
        
    Returns:
        tuple: (is_valid, list of issues)
    """
    issues = []
    
    # Check for missing values
    if data.isnull().any():
        issues.append("Data contains missing values")
    
    # Check for negative prices
    if (data < 0).any():
        issues.append("Data contains negative prices")
    
    # Check for sufficient data points
    if len(data) < 30:
        issues.append("Insufficient data points (minimum 30 required)")
    
    # Check for extreme values
    mean = data.mean()
    std = data.std()
    if ((data - mean).abs() > 5 * std).any():
        issues.append("Data contains extreme outliers")
    
    return len(issues) == 0, issues

def show_statistics(price_paths: List[List[float]], vol_paths: List[List[float]], bs_prices: List[float]):
    """Display simulation statistics.
    
    Args:
        price_paths: List of IG-OU price paths
        vol_paths: List of volatility paths
        bs_prices: List of Black-Scholes prices
    """
    # Convert to numpy arrays
    price_paths = np.array(price_paths)
    vol_paths = np.array(vol_paths)
    bs_prices = np.array(bs_prices)
    
    # Calculate final price statistics
    final_igou_prices = price_paths[:, -1]
    final_bs_price = bs_prices[-1]
    
    igou_stats = {
        'Mean': np.mean(final_igou_prices),
        'Std Dev': np.std(final_igou_prices),
        'Min': np.min(final_igou_prices),
        'Max': np.max(final_igou_prices),
        '5th Percentile': np.percentile(final_igou_prices, 5),
        '95th Percentile': np.percentile(final_igou_prices, 95)
    }
    
    # Calculate volatility statistics
    final_vols = vol_paths[:, -1]
    vol_stats = {
        'Mean': np.mean(final_vols),
        'Std Dev': np.std(final_vols),
        'Min': np.min(final_vols),
        'Max': np.max(final_vols),
        '5th Percentile': np.percentile(final_vols, 5),
        '95th Percentile': np.percentile(final_vols, 95)
    }
    
    # Display statistics
    st.subheader("Simulation Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("IG-OU Final Price Statistics")
        st.dataframe(pd.DataFrame({
            'Metric': list(igou_stats.keys()),
            'Value': [f"{v:.2f}" for v in igou_stats.values()]
        }).set_index('Metric'))
        
        st.write(f"Black-Scholes Final Price: {final_bs_price:.2f}")
    
    with col2:
        st.write("IG-OU Volatility Statistics")
        st.dataframe(pd.DataFrame({
            'Metric': list(vol_stats.keys()),
            'Value': [f"{v:.4f}" for v in vol_stats.values()]
        }).set_index('Metric'))
