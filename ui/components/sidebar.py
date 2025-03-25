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
        "Sélectionnez l'actif",
        options=VALID_ASSETS,
        help="Choose the asset to analyze"
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Sélectionnez un intervalle de temps",
        options=VALID_TIMEFRAMES,
        help="Choose the data timeframe"
    )
    
    # Model parameters
    st.sidebar.subheader("IG-OU Parameters")
    
    n_simulations = st.sidebar.number_input(
        "Simulations de Monte Carlo",
        min_value=1,
        max_value=1000,
        value=100
    )
    
    # Note explicative sur les paramètres a et b
    st.sidebar.info("""
    **Note sur les paramètres a et b:**
    
    Les paramètres a et b de la distribution Inverse Gaussienne sont calculés à partir des formules (3.16) de l'article WCE 2009, une fois que les donées sont chargées.  
    C'est pourquoi ils ne peuvent pas être saisis manuellement. 
    Ils apparaîtront automatiquement ci-dessous. 
    """)
    
    # Afficher les paramètres a et b calculés s'ils existent
    if 'ig_param_a' in st.session_state and 'ig_param_b' in st.session_state:
        st.sidebar.subheader("Paramètres calculés")
        st.sidebar.metric("Paramètre a", f"{st.session_state['ig_param_a']:.10f}")
        st.sidebar.metric("Paramètre b", f"{st.session_state['ig_param_b']:.10f}")
    
    return {
        "asset": asset,
        "timeframe": timeframe,
        "n_simulations": n_simulations
    }
