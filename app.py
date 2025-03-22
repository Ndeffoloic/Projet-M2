"""Main application module for the IG-OU asset price prediction."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.data.loader import load_asset_data
from core.estimators.parameters import ParameterEstimator
from core.models.black_scholes import BlackScholesModel
from core.models.bns import BNSModel
from core.models.ig_ou import IGOUModel
from ui.components.sidebar import render_sidebar
from ui.components.visualizations import plot_predictions, plot_volatility, VolatilityPlotter
from ui.helpers import show_statistics


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Prédiction Prix & Volatilité",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Prédiction de Prix des Actifs avec Modèles IG-OU et BNS")
    
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
    if config["timeframe"] in ["minute", "hour"] or config['asset'] == "GLE.PA":
        # Vérifier si price_series est scalaire ou itérable
        if np.isscalar(price_series) or isinstance(price_series, (int, float)):
            price_df = pd.DataFrame({'Close': [price_series]}, index=[pd.Timestamp.now()])
        else:
            price_df = pd.DataFrame({'Close': price_series})
    else:
        # Même chose pour l'autre branche
        if np.isscalar(price_series) or isinstance(price_series, (int, float)):
            price_df = pd.DataFrame({'Price': [price_series]}, index=[pd.Timestamp.now()])
        else:
            price_df = pd.DataFrame({'Price': price_series})
    
    # Estimate parameters
    mu, sigma_sq, lambda_ = ParameterEstimator.estimate_igou_parameters(price_df)
    bs_mu, bs_sigma = ParameterEstimator.estimate_bs_parameters(price_df)
    
    # Déterminer quelle colonne de prix est disponible
    price_col = 'Close' if 'Close' in price_df.columns else 'Price'
    a, b = ParameterEstimator.estimate_ig_ab(price_df[price_col], lambda_)
    
    # Initialize models
    igou_model = IGOUModel(lambda_=lambda_, a=a, b=b)
    bs_model = BlackScholesModel(bs_mu, bs_sigma)
    bns_model = BNSModel(igou_model, mu=mu, beta=0.5)  # Beta from paper
    
    # Run simulations
    if st.button("Lancer la simulation"):
        if len(price_series) > 0:
            last_price = price_series.iloc[-1]
            init_vol = np.sqrt(sigma_sq)
            
            # Exécuter les simulations de tous les modèles et générer les visualisations
            run_simulations(last_price, igou_model, bs_model, bns_model, config["n_simulations"], init_vol, len(price_series))
            
            # Calculer les rendements des différents modèles pour les diagnostics
            returns = price_series.pct_change().dropna()
            
            # Simuler tous les modèles pour le graphique de diagnostic
            igou_prices = []
            bs_prices = []
            bns_prices = []
            vol_series = []
            
            # Utiliser toute la longueur des données historiques
            T = len(price_series)
            
            # Effectuer plusieurs simulations pour chaque modèle
            for _ in range(5):  # Quelques simulations pour les diagnostics
                # IG-OU simulation
                vol_path = igou_model.simulate(X0=init_vol, T=T)
                price_path = [last_price]
                for vol in vol_path[1:]:  # Skip first vol since we use it for the first step
                    price_path.append(price_path[-1] * (1 + np.random.normal(0, vol)))
                igou_prices.extend(price_path)
                
                # Black-Scholes simulation
                bs_path = bs_model.simulate(S0=last_price, days=T)
                bs_prices.extend(bs_path)
                
                # BNS simulation
                bns_path, bns_vol = bns_model.simulate(last_price, T=T)
                bns_prices.extend(bns_path)
                vol_series.extend(bns_vol)
            
            # Convertir en séries pour le diagnostic
            igou_returns = pd.Series(igou_prices).pct_change().dropna()
            bs_returns = pd.Series(bs_prices).pct_change().dropna()
            bns_returns = pd.Series(bns_prices).pct_change().dropna()
            
            # Créer trois rubriques distinctes de diagnostics (une pour chaque modèle)
            st.header("Diagnostic du modèle Black-Scholes")
            st.pyplot(VolatilityPlotter.plot_diagnostics(
                returns=returns,
                model_returns=bs_returns,
                vol_series=pd.Series([bs_sigma] * len(vol_path)),  # Volatilité constante
                model_name="Black-Scholes"
            ))
            
            st.header("Diagnostic du modèle IG-OU")
            st.pyplot(VolatilityPlotter.plot_diagnostics(
                returns=returns,
                model_returns=igou_returns,
                vol_series=pd.Series(vol_path),
                model_name="IG-OU"
            ))
            
            st.header("Diagnostic du modèle BNS")
            st.pyplot(VolatilityPlotter.plot_diagnostics(
                returns=returns,
                model_returns=bns_returns,
                vol_series=pd.Series(vol_series),
                model_name="BNS"
            ))
        else:
            st.error("Données insuffisantes pour la simulation")

def run_simulations(last_price: float, igou_model: IGOUModel, bs_model: BlackScholesModel, 
                   bns_model: BNSModel, n_simulations: int, init_vol: float, T: int):
    """Execute and display simulations.
    
    Args:
        last_price: Last observed price
        igou_model: IG-OU model instance
        bs_model: Black-Scholes model instance
        bns_model: BNS model instance
        n_simulations: Number of simulations to run
        init_vol: Initial volatility
        T: Total number of time steps
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Listes pour stocker les trajectoires de prix et de volatilité
    igou_price_paths = []
    igou_vol_paths = []
    bs_price_paths = []
    bns_price_paths = []
    bns_vol_paths = []
    
    # Exécuter les simulations pour chaque modèle
    for _ in range(n_simulations):
        # Simulation IG-OU
        vol_path = igou_model.simulate(X0=init_vol, T=T)
        igou_vol_paths.append(vol_path)
        
        # Simulation des prix IG-OU
        price_path = [last_price]
        for vol in vol_path[1:]:  # Skip first vol since we use it for the first step
            price_path.append(price_path[-1] * (1 + np.random.normal(0, vol)))
        igou_price_paths.append(price_path)
        
        # Simulation Black-Scholes 
        bs_path = bs_model.simulate(S0=last_price, days=T)
        bs_price_paths.append(bs_path)
        
        # Simulation BNS
        bns_prices, bns_vol = bns_model.simulate(last_price, T=T)
        bns_price_paths.append(bns_prices)
        bns_vol_paths.append(bns_vol)
    
    # Plot predictions pour les trois modèles
    plot_all_predictions(ax1, igou_price_paths, bs_price_paths, bns_price_paths)
    
    # Plot volatility pour les deux modèles avec volatilité stochastique
    plot_all_volatility(ax2, igou_vol_paths, bns_vol_paths)
    
    # Affichage des graphiques
    st.pyplot(fig)

def plot_all_predictions(ax, igou_paths, bs_paths, bns_paths):
    """Tracer les prédictions de prix pour les trois modèles.
    
    Args:
        ax: Axe matplotlib pour le tracé
        igou_paths: Trajectoires des prix du modèle IG-OU
        bs_paths: Trajectoires des prix du modèle Black-Scholes
        bns_paths: Trajectoires des prix du modèle BNS
    """
    # Vérifier que les chemins ne sont pas vides
    if not igou_paths or not bs_paths or not bns_paths:
        ax.text(0.5, 0.5, "Données de prix insuffisantes", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Vérifier et standardiser les longueurs des séries
    lengths_igou = [len(path) for path in igou_paths]
    lengths_bs = [len(path) for path in bs_paths]
    lengths_bns = [len(path) for path in bns_paths]
    
    # Utiliser la longueur minimale pour tous
    min_length = min(min(lengths_igou), min(lengths_bs), min(lengths_bns))
    
    # Tronquer à la longueur minimale commune pour éviter les problèmes
    igou_paths_trimmed = [path[:min_length] for path in igou_paths]
    bs_paths_trimmed = [path[:min_length] for path in bs_paths]
    bns_paths_trimmed = [path[:min_length] for path in bns_paths]
    
    # Conversion en tableaux numpy
    igou_paths_np = np.array(igou_paths_trimmed)
    bs_paths_np = np.array(bs_paths_trimmed)
    bns_paths_np = np.array(bns_paths_trimmed)
    
    # Calcul des statistiques pour IG-OU
    mean_igou = np.mean(igou_paths_np, axis=0)
    p05_igou = np.percentile(igou_paths_np, 5, axis=0)
    p95_igou = np.percentile(igou_paths_np, 95, axis=0)
    
    # Calcul des statistiques pour Black-Scholes
    mean_bs = np.mean(bs_paths_np, axis=0)
    p05_bs = np.percentile(bs_paths_np, 5, axis=0)
    p95_bs = np.percentile(bs_paths_np, 95, axis=0)
    
    # Calcul des statistiques pour BNS
    mean_bns = np.mean(bns_paths_np, axis=0)
    p05_bns = np.percentile(bns_paths_np, 5, axis=0)
    p95_bns = np.percentile(bns_paths_np, 95, axis=0)
    
    # Création de l'axe X
    x_axis = np.arange(len(mean_igou))
    
    # Tracer IG-OU
    ax.plot(x_axis, mean_igou, color='blue', label='IG-OU (Moyenne)')
    ax.fill_between(x_axis, p05_igou, p95_igou, color='blue', alpha=0.1, label='IG-OU (90% IC)')
    
    # Tracer Black-Scholes
    ax.plot(x_axis, mean_bs, color='green', linestyle='--', label='Black-Scholes (Moyenne)')
    ax.fill_between(x_axis, p05_bs, p95_bs, color='green', alpha=0.1, label='BS (90% IC)')
    
    # Tracer BNS
    ax.plot(x_axis, mean_bns, color='red', linestyle='-.', label='BNS (Moyenne)')
    ax.fill_between(x_axis, p05_bns, p95_bns, color='red', alpha=0.1, label='BNS (90% IC)')
    
    # Personnalisation du graphique
    ax.set_title('Prédictions de prix selon les trois modèles', fontsize=14)
    ax.set_xlabel('Jours', fontsize=12)
    ax.set_ylabel('Prix', fontsize=12)
    ax.legend()
    ax.grid(True)

def plot_all_volatility(ax, igou_vol_paths, bns_vol_paths):
    """Tracer les trajectoires de volatilité pour les modèles avec volatilité stochastique.
    
    Args:
        ax: Axe matplotlib pour le tracé
        igou_vol_paths: Trajectoires de volatilité du modèle IG-OU
        bns_vol_paths: Trajectoires de volatilité du modèle BNS
    """
    # Vérifier que les chemins ne sont pas vides
    if not igou_vol_paths or not bns_vol_paths:
        ax.text(0.5, 0.5, "Données de volatilité insuffisantes", 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    # Vérifier et standardiser les longueurs des séries
    lengths_igou = [len(path) for path in igou_vol_paths]
    lengths_bns = [len(path) for path in bns_vol_paths]
    
    # Utiliser la longueur minimale pour tous
    min_length = min(min(lengths_igou), min(lengths_bns))
    
    # Tronquer à la longueur minimale commune pour éviter les problèmes
    igou_vol_paths_trimmed = [path[:min_length] for path in igou_vol_paths]
    bns_vol_paths_trimmed = [path[:min_length] for path in bns_vol_paths]
    
    # Conversion en tableaux numpy avec dimensions correctes
    igou_vol_paths_np = np.array(igou_vol_paths_trimmed)
    bns_vol_paths_np = np.array(bns_vol_paths_trimmed)
    
    # Protection contre les données aberrantes
    igou_vol_paths_np = np.clip(igou_vol_paths_np, 0, 2.0)
    bns_vol_paths_np = np.clip(bns_vol_paths_np, 0, 2.0)
    
    # Calcul des statistiques pour IG-OU
    mean_igou_vol = np.mean(igou_vol_paths_np, axis=0)
    p05_igou_vol = np.percentile(igou_vol_paths_np, 5, axis=0)
    p95_igou_vol = np.percentile(igou_vol_paths_np, 95, axis=0)
    
    # Calcul des statistiques pour BNS
    mean_bns_vol = np.mean(bns_vol_paths_np, axis=0)
    p05_bns_vol = np.percentile(bns_vol_paths_np, 5, axis=0)
    p95_bns_vol = np.percentile(bns_vol_paths_np, 95, axis=0)
    
    # Tracer IG-OU
    ax.plot(mean_igou_vol, color='blue', label='IG-OU (Moyenne)')
    ax.fill_between(range(len(mean_igou_vol)), p05_igou_vol, p95_igou_vol, color='blue', alpha=0.1, label='IG-OU (90% IC)')
    
    # Tracer BNS
    ax.plot(mean_bns_vol, color='red', linestyle='-.', label='BNS (Moyenne)')
    ax.fill_between(range(len(mean_bns_vol)), p05_bns_vol, p95_bns_vol, color='red', alpha=0.1, label='BNS (90% IC)')
    
    # Personnalisation du graphique
    ax.set_title('Volatilité des modèles IG-OU et BNS', fontsize=14)
    ax.set_xlabel('Jours', fontsize=12)
    ax.set_ylabel('Volatilité', fontsize=12)
    ax.legend()
    ax.grid(True)

if __name__ == "__main__":
    main()
