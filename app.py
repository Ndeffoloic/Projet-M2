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
from ui.components.visualizations import (VolatilityPlotter, plot_predictions,
                                          plot_volatility)
from ui.helpers import calculate_descriptive_stats, show_statistics


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Application des méthodes de l'article WCE2009 ",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Estimation Bayésienne de Modèles à Volatilité Stochastique")
    
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
    
    # Estimate parameters with improved method
    mu, sigma_sq, lambda_, a, b = ParameterEstimator.estimate_igou_parameters(price_df)
    bs_mu, bs_sigma = ParameterEstimator.estimate_bs_parameters(price_df)
    
    # Stocker les paramètres a et b dans la session state pour affichage dans la sidebar
    st.session_state['ig_param_a'] = a
    st.session_state['ig_param_b'] = b
    
    # Initialize models
    igou_model = IGOUModel(lambda_=lambda_, a=a, b=b)
    bs_model = BlackScholesModel(bs_mu, bs_sigma)
    
    # Initialiser le modèle BNS avec la nouvelle implémentation
    # Utiliser un paramètre de levier (rho) négatif pour capturer l'effet de levier
    bns_model = BNSModel(lambda_=lambda_, a=a, b=b, mu=mu, rho=-0.5)
    
    # Run simulations
    if st.button("Lancer la simulation"):
        if len(price_series) > 0:
            last_price = price_series.iloc[-1]
            init_vol = np.sqrt(sigma_sq)
            
            # Exécuter les simulations de tous les modèles et générer les visualisations
            bs_price_paths, igou_price_paths, igou_vol_paths, bns_price_paths, bns_vol_paths = run_simulations(
                last_price, igou_model, bs_model, bns_model, config["n_simulations"], init_vol, len(price_series)
            )
            
            # Calculer les rendements à partir des prix simulés
            # Prendre la première simulation de chaque modèle pour les comparaisons
            bs_prices = pd.Series(bs_price_paths[0])
            igou_prices = pd.Series(igou_price_paths[0])
            bns_prices = pd.Series(bns_price_paths[0])
            
            # Calculer les rendements logarithmiques
            bs_returns = np.log(bs_prices / bs_prices.shift(1)).dropna()
            igou_returns = np.log(igou_prices / igou_prices.shift(1)).dropna()
            bns_returns = np.log(bns_prices / bns_prices.shift(1)).dropna()
            
            returns = price_series.pct_change().dropna()
            
            # Nouvelles visualisations selon WCE 2009
            st.header("Analyse Comparative selon WCE 2009")
            
            # Figure 2: Autocorrélation
            st.subheader("Autocorrélation réelle vs estimée")
            fig_autocorr = VolatilityPlotter.plot_autocorrelation_comparison(
                actual_returns=returns,
                igou_returns=igou_returns,
                bs_returns=bs_returns,
                bns_returns=bns_returns
            )
            st.pyplot(fig_autocorr)
            
            # Figure 6: Rendements
            st.subheader("Comparaison des rendements historiques et estimés")
            fig_returns = VolatilityPlotter.plot_returns_comparison(
                actual_returns=returns.values,
                igou_returns=igou_returns,
                bs_returns=bs_returns,
                bns_returns=bns_returns
            )
            st.pyplot(fig_returns)
            
            # Diagnostics spécifiques à chaque modèle
            show_bs = True
            show_igou = True
            show_bns = True
            
            if show_bs:
                st.header("Diagnostic du modèle Black-Scholes")
                bs_diag = VolatilityPlotter.plot_diagnostics(returns, bs_returns, None, "Black-Scholes")
                st.pyplot(bs_diag)
                
                # ACF des résidus carrés pour Black-Scholes
                # st.subheader("ACF des résidus carrés pour Black-Scholes")
                # bs_acf = VolatilityPlotter.plot_residuals_acf(returns, bs_returns, "Black-Scholes")
                # st.pyplot(bs_acf)
            
            if show_igou:
                st.header("Diagnostic du modèle IG-OU")
                igou_diag = VolatilityPlotter.plot_diagnostics(returns, igou_returns, igou_vol_paths, "IG-OU")
                st.pyplot(igou_diag)
                
                # ACF des résidus carrés pour IG-OU
                # st.subheader("ACF des résidus carrés pour IG-OU")
                # igou_acf = VolatilityPlotter.plot_residuals_acf(returns, igou_returns, "IG-OU")
                # st.pyplot(igou_acf)
            
            if show_bns:
                st.header("Diagnostic du modèle BNS")
                bns_diag = VolatilityPlotter.plot_diagnostics(returns, bns_returns, bns_vol_paths, "BNS")
                st.pyplot(bns_diag)
                
                # ACF des résidus carrés pour BNS
                # st.subheader("ACF des résidus carrés pour BNS")
                # bns_acf = VolatilityPlotter.plot_residuals_acf(returns, bns_returns, "BNS")
                # st.pyplot(bns_acf)
                
            # Tableau comparatif des statistiques descriptives (Table 1)
            st.header("Comparaison des statistiques descriptives")
            
            # Calculer les statistiques pour chaque série
            actual_stats = calculate_descriptive_stats(returns, "Actual Returns")
            bs_stats = calculate_descriptive_stats(pd.Series(bs_returns), "Black-Scholes")
            igou_stats = calculate_descriptive_stats(pd.Series(igou_returns), "IG-OU")
            bns_stats = calculate_descriptive_stats(pd.Series(bns_returns), "BNS")
            
            # Fusionner les dataframes
            combined_stats = pd.concat([
                actual_stats, 
                bs_stats,
                igou_stats,
                bns_stats
            ], axis=1)
            
            # Afficher le tableau avec mise en forme
            st.dataframe(
                combined_stats.style.format("{:.4f}")
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('text-align', 'center')]
                }]),
                height=400
            )
            
            # Explication des métriques
            with st.expander("Explication des métriques"):
                st.markdown("""
                - **Sample Size** : Nombre d'observations
                - **Range** : Différence entre les valeurs max/min
                - **Coef. of Variation** : Ratio écart-type/moyenne (mesure de dispersion relative)
                - **Excess Kurtosis** : Kurtosis - 3 (0 = distribution normale)
                """)
            
            # Graphique comparatif des ACF des résidus carrés
            st.header("Comparaison des ACF des résidus carrés")
            fig_acf_comp, ax_acf_comp = plt.subplots(figsize=(15, 8))
            plot_residuals_comparison(ax_acf_comp, returns, igou_returns, bs_returns, bns_returns)
            st.pyplot(fig_acf_comp)
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
        
    Returns:
        Tuple containing lists of price paths and volatility paths for each model
    """
    
    # Listes pour stocker les trajectoires de prix et de volatilité
    igou_price_paths = []
    igou_vol_paths = []
    bs_price_paths = []
    bns_price_paths = []
    bns_vol_paths = []
    
    # Exécuter les simulations pour chaque modèle
    for _ in range(n_simulations):
        try:
            # Simulation IG-OU
            vol_path = igou_model.simulate(X0=init_vol, T=T)
            igou_vol_paths.append(vol_path)
            
            # Simulation des prix IG-OU
            price_path = [last_price]
            for vol in vol_path[1:]:  # Skip first vol since we use it for the first step
                # Limiter la volatilité pour éviter les explosions numériques
                vol_capped = min(vol, 0.5)
                try:
                    next_price = price_path[-1] * (1 + np.random.normal(0, vol_capped))
                    # Si le prix devient négatif ou NaN, utiliser une approche plus conservative
                    if next_price <= 0 or np.isnan(next_price) or np.isinf(next_price):
                        next_price = price_path[-1] * (1 + np.random.normal(0, 0.01))
                except:
                    # En cas d'erreur, utiliser une mise à jour plus simple
                    next_price = price_path[-1] * 1.001
                price_path.append(next_price)
            igou_price_paths.append(price_path)
            
            # Simulation Black-Scholes 
            bs_path = bs_model.simulate(S0=last_price, days=T)
            bs_price_paths.append(bs_path)
            
            # Simulation BNS
            bns_prices, bns_vol = bns_model.simulate(last_price, T=T)
            bns_price_paths.append(bns_prices)
            bns_vol_paths.append(bns_vol)
        except Exception as e:
            # En cas d'erreur, log et continuer
            print(f"Erreur lors de la simulation {_}: {str(e)}")
            # Ajouter des chemins vides pour maintenir la cohérence
            if len(igou_price_paths) < _ + 1:
                igou_price_paths.append([last_price] * T)
            if len(igou_vol_paths) < _ + 1:
                igou_vol_paths.append([init_vol] * T)
            if len(bs_price_paths) < _ + 1:
                bs_price_paths.append([last_price] * T)
            if len(bns_price_paths) < _ + 1:
                bns_price_paths.append([last_price] * T)
            if len(bns_vol_paths) < _ + 1:
                bns_vol_paths.append([init_vol] * T)
    
    # S'assurer qu'il y a au moins une simulation pour chaque modèle
    if not igou_price_paths:
        igou_price_paths.append([last_price] * T)
    if not igou_vol_paths:
        igou_vol_paths.append([init_vol] * T)
    if not bs_price_paths:
        bs_price_paths.append([last_price] * T)
    if not bns_price_paths:
        bns_price_paths.append([last_price] * T)
    if not bns_vol_paths:
        bns_vol_paths.append([init_vol] * T)
    
    return bs_price_paths, igou_price_paths, igou_vol_paths, bns_price_paths, bns_vol_paths

def plot_all_predictions(ax, bs_paths, igou_paths, bns_paths, price_for_plot):
    """Tracer les prédictions de prix pour les trois modèles.
    
    Args:
        ax: Axe matplotlib pour le tracé
        bs_paths: Trajectoires des prix du modèle Black-Scholes
        igou_paths: Trajectoires des prix du modèle IG-OU
        bns_paths: Trajectoires des prix du modèle BNS
        price_for_plot: Prix pour la ligne de référence
    """
    # Vérifier que les chemins ne sont pas vides
    if not bs_paths or not igou_paths or not bns_paths:
        ax.text(0.5, 0.5, "Données de prix insuffisantes", 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    # Vérifier et standardiser les longueurs des séries
    lengths_bs = [len(path) for path in bs_paths]
    lengths_igou = [len(path) for path in igou_paths]
    lengths_bns = [len(path) for path in bns_paths]
    
    # Utiliser la longueur minimale pour tous
    min_length = min(min(lengths_bs), min(lengths_igou), min(lengths_bns))
    
    # Tronquer à la longueur minimale commune pour éviter les problèmes
    bs_paths_trimmed = [path[:min_length] for path in bs_paths]
    igou_paths_trimmed = [path[:min_length] for path in igou_paths]
    bns_paths_trimmed = [path[:min_length] for path in bns_paths]
    
    # Conversion en tableaux numpy
    bs_paths_np = np.array(bs_paths_trimmed)
    igou_paths_np = np.array(igou_paths_trimmed)
    bns_paths_np = np.array(bns_paths_trimmed)
    
    # Calcul des statistiques pour Black-Scholes
    mean_bs = np.mean(bs_paths_np, axis=0)
    p05_bs = np.percentile(bs_paths_np, 5, axis=0)
    p95_bs = np.percentile(bs_paths_np, 95, axis=0)
    
    # Calcul des statistiques pour IG-OU
    mean_igou = np.mean(igou_paths_np, axis=0)
    p05_igou = np.percentile(igou_paths_np, 5, axis=0)
    p95_igou = np.percentile(igou_paths_np, 95, axis=0)
    
    # Calcul des statistiques pour BNS
    mean_bns = np.mean(bns_paths_np, axis=0)
    p05_bns = np.percentile(bns_paths_np, 5, axis=0)
    p95_bns = np.percentile(bns_paths_np, 95, axis=0)
    
    # Création de l'axe X
    x_axis = np.arange(len(mean_bs))
    
    # Tracer Black-Scholes
    ax.plot(x_axis, mean_bs, color='green', linestyle='--', label='Black-Scholes (Moyenne)')
    ax.fill_between(x_axis, p05_bs, p95_bs, color='green', alpha=0.1, label='Black-Scholes (90% IC)')
    
    # Tracer IG-OU
    ax.plot(x_axis, mean_igou, color='blue', label='IG-OU (Moyenne)')
    ax.fill_between(x_axis, p05_igou, p95_igou, color='blue', alpha=0.1, label='IG-OU (90% IC)')
    
    # Tracer BNS
    ax.plot(x_axis, mean_bns, color='red', linestyle='-.', label='BNS (Moyenne)')
    ax.fill_between(x_axis, p05_bns, p95_bns, color='red', alpha=0.1, label='BNS (90% IC)')
    
    # Ligne de référence pour le prix actuel
    # Convertir price_for_plot en float si c'est une série
    if isinstance(price_for_plot, pd.Series):
        price_value = float(price_for_plot.iloc[-1])
    else:
        price_value = float(price_for_plot)
    
    ax.axhline(y=price_value, color='black', linestyle='-', label='Prix actuel')
    
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
    if igou_vol_paths is None or len(igou_vol_paths) == 0:
        ax.text(0.5, 0.5, "Données de volatilité IG-OU non disponibles", ha='center', va='center', transform=ax.transAxes)
        return
        
    if bns_vol_paths is None or len(bns_vol_paths) == 0:
        ax.text(0.5, 0.5, "Données de volatilité BNS non disponibles", ha='center', va='center', transform=ax.transAxes)
        return
        
    # Trouver la longueur minimale parmi les chemins pour l'alignement
    min_len = min(len(igou_vol_paths), len(bns_vol_paths))

    # Créer l'axe temporel
    x_axis = np.arange(min_len)
    
    # Tracer les trajectoires de volatilité
    ax.plot(x_axis, igou_vol_paths[:min_len], 'r--', label='IG-OU', alpha=0.7)
    ax.plot(x_axis, bns_vol_paths[:min_len], 'g-', label='BNS', alpha=0.7)
    
    # Personnalisation du graphique
    ax.set_title('Comparaison des volatilités stochastiques', fontsize=14)
    ax.set_xlabel('Jours', fontsize=12)
    ax.set_ylabel('Volatilité', fontsize=12)
    ax.legend()
    ax.grid(True)

def plot_residuals_comparison(ax, actual_returns, igou_returns, bs_returns, bns_returns):
    """Graphique de comparaison des ACF des résidus carrés pour tous les modèles.
    
    Args:
        ax: Axe matplotlib pour le tracé
        actual_returns: Rendements réels observés
        igou_returns: Rendements simulés par le modèle IG-OU
        bs_returns: Rendements simulés par le modèle Black-Scholes
        bns_returns: Rendements simulés par le modèle BNS
    """
    # S'assurer que les données sont sous le bon format
    def ensure_series(data):
        if not isinstance(data, pd.Series):
            return pd.Series(data)
        return data
        
    actual_returns = ensure_series(actual_returns)
    igou_returns = ensure_series(igou_returns)
    bs_returns = ensure_series(bs_returns)
    bns_returns = ensure_series(bns_returns)
    
    # Réinitialiser les index pour éviter les problèmes de comparaison
    actual_returns = actual_returns.reset_index(drop=True)
    igou_returns = igou_returns.reset_index(drop=True)
    bs_returns = bs_returns.reset_index(drop=True)
    bns_returns = bns_returns.reset_index(drop=True)
    
    # Déterminer la longueur minimale pour calculer les résidus
    min_length = min(len(actual_returns), len(igou_returns), len(bs_returns), len(bns_returns))
    
    # Si les données sont insuffisantes
    if min_length < 5:
        ax.text(0.5, 0.5, "Données insuffisantes pour calculer les ACF des résidus (minimum 5 points nécessaires)", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    try:
        # Calculer les résidus carrés pour chaque modèle en utilisant les valeurs numpy
        res_igou = (actual_returns.values[:min_length] - igou_returns.values[:min_length])**2
        res_bs = (actual_returns.values[:min_length] - bs_returns.values[:min_length])**2
        res_bns = (actual_returns.values[:min_length] - bns_returns.values[:min_length])**2
        
        # Calculer les autocorrélations
        from statsmodels.tsa.stattools import acf
        
        max_lags = min(40, min_length // 2)
        acf_igou = acf(res_igou, nlags=max_lags, fft=True)
        acf_bs = acf(res_bs, nlags=max_lags, fft=True)
        acf_bns = acf(res_bns, nlags=max_lags, fft=True)
        
        # Créer l'axe des x
        lags_x = range(len(acf_igou))
        
        # Tracer les autocorrélations des résidus carrés
        ax.plot(lags_x, acf_igou, 'r--', label='IG-OU', alpha=0.7)
        ax.plot(lags_x, acf_bs, 'g-.', label='Black-Scholes', alpha=0.7)
        ax.plot(lags_x, acf_bns, 'm:', label='BNS', alpha=0.7, linewidth=2)
        
        ax.set_title("ACF des résidus carrés selon les modèles", fontsize=14)
        ax.set_xlabel("Retards (lags)", fontsize=12)
        ax.set_ylabel("Autocorrélation", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=12)
    except Exception as e:
        ax.text(0.5, 0.5, f"Erreur lors du calcul des ACF: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)

if __name__ == "__main__":
    main()
