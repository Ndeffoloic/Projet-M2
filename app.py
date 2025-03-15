"""Main application module for the IG-OU asset price prediction."""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    
    # Convert price series to DataFrame with 'Close' column
    price_df = pd.DataFrame({'Close': price_series})
    
    # Estimate parameters
    mu, sigma_sq, lambda_ = ParameterEstimator.estimate_igou_parameters(price_df)
    bs_mu, bs_sigma = ParameterEstimator.estimate_bs_parameters(price_df)
    
    # Initialize models
    igou_model = IGOUModel(lambda_, config["a"], config["b"])
    bs_model = BlackScholesModel(bs_mu, bs_sigma)
    
    # Run simulations
    if st.button("Lancer la simulation"):
        if len(price_series) > 0:
            last_price = price_series.iloc[-1]
            init_vol = np.sqrt(sigma_sq)
            run_simulations(last_price, igou_model, bs_model, config["n_simulations"], init_vol)
        else:
            st.error("Données insuffisantes pour la simulation")

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
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Run IG-OU simulations
    price_paths = []
    vol_paths = []
    for _ in range(n_simulations):
        vol_path = igou_model.simulate(X0=init_vol, T=30)
        vol_paths.append(vol_path)
        
        # Simulate price path using the volatility path
        price_path = [last_price]
        for vol in vol_path[1:]:  # Skip first vol since we use it for the first step
            price_path.append(price_path[-1] * (1 + np.random.normal(0, vol)))
        price_paths.append(price_path)
    
    # Run Black-Scholes simulation
    bs_path = bs_model.simulate(S0=last_price, days=30)
    
    # Plot predictions and volatility
    plot_predictions(ax1, price_paths, bs_path)
    plot_volatility(ax2, vol_paths)
    
    # Display plots
    st.pyplot(fig)
    
    # Show statistics
    show_statistics(price_paths, vol_paths, bs_path)

if __name__ == "__main__":
    main()
