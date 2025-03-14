import streamlit as st
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# Import des composants core
from core.models.ig_ou import IGOUProcess
from core.models.black_scholes import BSModel
from core.estimators.parameters import IGParameterEstimator, BSParameterEstimator
from core.data.loader import AssetLoader

# Import des composants UI
from ui.components.sidebar import Sidebar
from ui.components.visualizations import VolatilityPlotter
from ui.helpers import (
    display_error,
    display_success,
    display_info,
    display_parameters,
    format_data_summary,
    check_data_quality
)

def load_data(config: dict) -> Optional[pd.Series]:
    """
    Charge les données selon la configuration
    
    Args:
        config: Configuration de l'application
        
    Returns:
        Optional[pd.Series]: Série temporelle des prix ou None si erreur
    """
    loader = AssetLoader()
    try:
        if config['source'] == 'yahoo':
            return loader.load_from_yahoo(config['symbol'], config['period'])
        elif config['source'] == 'file':
            if config['file'] is None:
                display_info("Veuillez télécharger un fichier")
                return None
            return loader.load_from_file(config['file'], config['date_column'], config['price_column'])
        else:  # example
            return loader.load_from_file(config['asset'], config['timeframe'])
    except Exception as e:
        display_error(f"Erreur lors du chargement des données: {str(e)}")
        return None

def run_simulations(
    prices: pd.Series,
    config: dict
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Exécute les simulations selon le modèle choisi
    
    Args:
        prices: Série temporelle des prix
        config: Configuration de l'application
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
            Trajectoires IG-OU et Black-Scholes (None si non calculées)
    """
    # Calcul des rendements
    returns = prices.pct_change().dropna()
    
    # Initialisation des résultats
    ig_paths = None
    bs_paths = None
    
    try:
        if config['model'] in ['IG-OU', 'Les deux']:
            # Estimation des paramètres IG-OU
            ig_params = IGParameterEstimator.estimate(returns)
            display_parameters(ig_params)
            
            # Simulation IG-OU
            model = IGOUProcess(
                lambda_=ig_params['lambda_'],
                a=np.sqrt(ig_params['sigma_sq']),
                b=1.0
            )
            
            ig_paths = np.array([
                model.simulate(returns.std(), config['horizon'])
                for _ in range(config['n_simulations'])
            ])
        
        if config['model'] in ['Black-Scholes', 'Les deux']:
            # Estimation des paramètres BS
            bs_params = BSParameterEstimator.estimate(returns)
            if config['model'] == 'Black-Scholes':
                display_parameters(bs_params)
            
            # Simulation Black-Scholes
            model = BSModel(mu=bs_params['mu'], sigma=bs_params['sigma'])
            bs_paths = np.array([
                model.simulate(prices.iloc[-1], config['horizon'])
                for _ in range(config['n_simulations'])
            ])
        
        return ig_paths, bs_paths
    
    except Exception as e:
        display_error(f"Erreur lors des simulations: {str(e)}")
        return None, None

def main():
    """Point d'entrée principal de l'application"""
    st.set_page_config(
        page_title="Prédiction Prix & Volatilité",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Prédiction de Prix et Volatilité")
    st.markdown("""
    Cette application implémente un modèle de volatilité stochastique IG-OU
    et le compare au modèle classique de Black-Scholes.
    """)
    
    # Chargement de la configuration
    config = Sidebar.render()
    
    # Chargement des données
    prices = load_data(config)
    if prices is None:
        return
    
    # Vérification de la qualité des données
    valid, issues = check_data_quality(prices)
    if not valid:
        st.warning("⚠️ Problèmes détectés dans les données:")
        for issue in issues:
            st.write(f"- {issue}")
    
    # Affichage du résumé des données
    format_data_summary(prices)
    
    # Affichage des données historiques
    returns = prices.pct_change().dropna()
    VolatilityPlotter.plot_historical_data(prices, returns)
    
    # Simulation et affichage des résultats
    ig_paths, bs_paths = run_simulations(prices, config)
    
    if ig_paths is not None or bs_paths is not None:
        st.subheader("Résultats des simulations")
        
        if config['model'] == 'IG-OU':
            VolatilityPlotter.plot_volatility_surface(
                ig_paths,
                np.arange(config['horizon']),
                "Surface de Volatilité IG-OU"
            )
        elif config['model'] == 'Les deux':
            VolatilityPlotter.plot_comparison(ig_paths, bs_paths)

if __name__ == "__main__":
    main()
