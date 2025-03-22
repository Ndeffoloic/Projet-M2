"""Sidebar component for parameter configuration."""
import streamlit as st

from core.data.loader import VALID_ASSETS, VALID_TIMEFRAMES

def render_sidebar() -> dict:
    """Render sidebar with model parameters.
    
    Returns:
        dict: Configuration parameters
    """
    st.sidebar.title("Model Parameters")
    
    # Asset selection
    asset = st.sidebar.selectbox(
        "Select Asset",
        options=VALID_ASSETS,
        help="Choose the asset to analyze"
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        options=VALID_TIMEFRAMES,
        help="Choose the data timeframe"
    )
    
    # Model parameters
    st.sidebar.subheader("IG-OU Parameters")
    
    n_simulations = st.sidebar.number_input(
        "Number of Simulations",
        min_value=1,
        max_value=1000,
        value=100,
        help="Number of Monte Carlo simulations"
    )
    
    # Note explicative sur les paramètres a et b
    st.sidebar.info("""
    **Note sur les paramètres a et b:**
    
    Selon l'article WCE 2009 (équation 3.16), les paramètres a et b de la distribution Inverse Gaussienne sont calculés automatiquement à partir de:
    - a = Ȳ * b / (e^(λh) - 1)
    - b = sqrt[(Var(Y) * (e^(2λh) - 1)) / (e^(λh) - 1)^3]
    
    Ces paramètres ne doivent pas être saisis manuellement.
    """)
    
    return {
        "asset": asset,
        "timeframe": timeframe,
        "n_simulations": n_simulations
    }
