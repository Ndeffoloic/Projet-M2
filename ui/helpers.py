from typing import List, Optional, Tuple, Union

import matplotlib as plt
import numpy as np
import pandas as pd
import streamlit as st


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
        
    def plot_diagnostics(returns: pd.Series, model_returns: pd.Series, vol_series: pd.Series):
        """Generate all diagnostic plots from the paper."""
        fig = plt.figure(figsize=(15, 20))
        
        # Autocorrelation plots
        ax1 = plt.subplot(321)
        plot_acf(returns, ax=ax1, title="Autocorrelation of Returns")
        
        ax2 = plt.subplot(322)
        plot_acf(vol_series, ax=ax2, title="Autocorrelation of Volatility")
        
        # PDF comparison
        ax3 = plt.subplot(323)
        sns.kdeplot(returns, ax=ax3, label="Empirical")
        x = np.linspace(returns.min(), returns.max(), 100)
        ax3.plot(x, invgauss.pdf(x, *inv_parameters), label="Inv Gaussian")
        ax3.set_title("Return Distribution vs IG")
        
        # Residual analysis
        ax4 = plt.subplot(324)
        residuals = returns - model_returns
        plot_acf(residuals**2, ax=ax4, title="ACF of Squared Residuals")
        
        # QQ Plot
        ax5 = plt.subplot(325)
        sm.qqplot(residuals, line='s', ax=ax5)
        ax5.set_title("QQ Plot vs Normal Distribution")
        
        # Volatility path
        ax6 = plt.subplot(326)
        vol_series.plot(ax=ax6, title="Simulated Volatility Path")
        
        plt.tight_layout()
        return fig
