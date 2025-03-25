"""Composant de barre latérale pour la configuration des paramètres."""
import streamlit as st

from core.data.loader import VALID_ASSETS, VALID_TIMEFRAMES


def render_sidebar() -> dict:
    """Affiche la barre latérale avec les paramètres du modèle.
    
    Returns:
        dict: Paramètres de configuration
    """
    st.sidebar.title("Paramètres du Modèle")
    
    # Sélection de l'actif
    actif = st.sidebar.selectbox(
        "Sélectionner l'Actif",
        options=VALID_ASSETS,
        help="Choisissez l'actif à analyser"
    )
    
    # Sélection de l'intervalle de temps
    intervalle_temps = st.sidebar.selectbox(
        "Sélectionner l'Intervalle de Temps",
        options=VALID_TIMEFRAMES,
        help="Choisissez l'intervalle de temps des données"
    )
    
    # Paramètres du modèle
    st.sidebar.subheader("Paramètres IG-OU")
    
    n_simulations = st.sidebar.number_input(
        "Nombre de Simulations",
        min_value=1,
        max_value=1000,
        value=100,
        help="Nombre de simulations Monte Carlo"
    )
    
    # Note explicative sur les paramètres a et b
    st.sidebar.info("""
    **Note sur les paramètres a et b:**
    
    Les paramètres a et b de la distribution Inverse Gaussienne sont calculés à partir des formules (3.16) de l'article WCE 2009, une fois que les données sont chargées.  
    C'est pourquoi ils ne peuvent pas être saisis manuellement. 
    Ils apparaîtront automatiquement ci-dessous. 
    """)
    
    # Afficher les paramètres a et b calculés s'ils existent
    if 'ig_param_a' in st.session_state and 'ig_param_b' in st.session_state:
        st.sidebar.subheader("Paramètres calculés")
        st.sidebar.metric("Paramètre a", f"{st.session_state['ig_param_a']:.10f}")
        st.sidebar.metric("Paramètre b", f"{st.session_state['ig_param_b']:.10f}")
    
    return {
        "actif": actif,
        "intervalle_temps": intervalle_temps,
        "n_simulations": n_simulations
    }
