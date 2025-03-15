"""Main application module for the IG-OU asset price prediction."""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core.data.loader import load_asset_data
from core.models.ig_ou import IGOUModel
from core.models.black_scholes import BlackScholesModel
from core.estimators.parameters import ParameterEstimator
from ui.components.sidebar import render_sidebar
from ui.components.visualizations import plot_predictions, plot_volatility
from ui.helpers import show_statistics

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Prédiction Prix & Volatilité",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Prédiction de Prix des Actifs avec Modèle IG-OU")
    
    # Get configuration from sidebar
    config = render_sidebar()
    
    # Load data
    price_series = load_asset_data(config["asset"], config["timeframe"])
    if price_series is None:
        return
    
    # Display historical data
    st.subheader(f"Données historiques - {config['asset']} ({config['timeframe']})")
    st.line_chart(price_series)
    
    # Calculate returns and estimate parameters
    returns = price_series.pct_change().dropna()
    mu, sigma_sq, lambda_ = ParameterEstimator.estimate_igou_parameters(returns)
    bs_mu, bs_sigma = ParameterEstimator.estimate_bs_parameters(returns)
    
    # Initialize models
    igou_model = IGOUModel(lambda_, config["a"], config["b"])
    bs_model = BlackScholesModel(bs_mu, bs_sigma)
    
    # Run simulations
    if st.button("Lancer la simulation"):
        run_simulations(
            price_series.iloc[-1],
            igou_model,
            bs_model,
            config["n_simulations"],
            returns.std()
        )

def run_simulations(last_price: float, igou_model: IGOUModel, bs_model: BlackScholesModel, 
                   n_simulations: int, init_vol: float):
    """Execute and display simulations.
    
    Args:
        last_price: Last observed price
        igou_model: IG-OU model instance
        bs_model: Black-Scholes model instance
        n_simulations: Number of simulations to run
        init_vol: Initial volatility
    """
    # Simulation IG-OU
    vol_paths = np.zeros((n_simulations, 30))
    price_paths = np.zeros((n_simulations, 30))
    
    for i in range(n_simulations):
        vol = igou_model.simulate(init_vol)
        prices = [last_price]
        for t in range(29):
            shock = vol[t] * np.random.normal()
            prices.append(prices[-1] * np.exp(igou_model.mu + shock))
        price_paths[i] = prices
        vol_paths[i] = vol
    
    # Simulation Black-Scholes
    bs_prices = bs_model.simulate(last_price)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    plot_predictions(ax1, price_paths, bs_prices)
    plot_volatility(ax2, vol_paths)
    st.pyplot(fig)
    
    # Show statistics
    show_statistics(price_paths, vol_paths, bs_prices)

if __name__ == "__main__":
    main()
