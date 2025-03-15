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
    
    a = st.sidebar.number_input(
        "Parameter a",
        min_value=1e-10,
        max_value=10.0,
        value=2.2395e-7,
        format="%.10f",
        help="IG-OU parameter a"
    )
    
    b = st.sidebar.number_input(
        "Parameter b",
        min_value=1e-10,
        max_value=10.0,
        value=1.0,
        help="IG-OU parameter b"
    )
    
    return {
        "asset": asset,
        "timeframe": timeframe,
        "n_simulations": n_simulations,
        "a": a,
        "b": b
    }
