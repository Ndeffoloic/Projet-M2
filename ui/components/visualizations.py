import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Optional

class VolatilityPlotter:
    """Génère les visualisations de volatilité selon les figures du document"""
    
    @staticmethod
    def plot_comparison(ig_paths: np.ndarray, bs_paths: np.ndarray, title: str = "Comparaison des prédictions de prix"):
        """
        Compare les trajectoires IG-OU et Black-Scholes
        
        Args:
            ig_paths: Trajectoires du modèle IG-OU (n_simulations x T)
            bs_paths: Trajectoires du modèle Black-Scholes (n_simulations x T)
            title: Titre du graphique
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot IG-OU avec intervalles de confiance
        mean_ig = np.mean(ig_paths, axis=0)
        p05_ig = np.percentile(ig_paths, 5, axis=0)
        p95_ig = np.percentile(ig_paths, 95, axis=0)
        
        ax.plot(mean_ig, color='red', label='IG-OU (Moyenne)')
        ax.fill_between(
            range(len(mean_ig)),
            p05_ig,
            p95_ig,
            color='red',
            alpha=0.1,
            label='IG-OU (90% IC)'
        )
        
        # Plot Black-Scholes avec intervalles de confiance
        mean_bs = np.mean(bs_paths, axis=0)
        p05_bs = np.percentile(bs_paths, 5, axis=0)
        p95_bs = np.percentile(bs_paths, 95, axis=0)
        
        ax.plot(mean_bs, color='blue', linestyle='--', label='Black-Scholes (Moyenne)')
        ax.fill_between(
            range(len(mean_bs)),
            p05_bs,
            p95_bs,
            color='blue',
            alpha=0.1,
            label='Black-Scholes (90% IC)'
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Périodes", fontsize=12)
        ax.set_ylabel("Prix", fontsize=12)
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        plt.close(fig)
    
    @staticmethod
    def plot_volatility_surface(volatilities: np.ndarray, times: np.ndarray, title: str = "Surface de Volatilité"):
        """
        Génère une surface 3D de volatilité
        
        Args:
            volatilities: Matrice de volatilités (n_simulations x T)
            times: Vecteur des temps
            title: Titre du graphique
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(times, range(volatilities.shape[0]))
        
        surf = ax.plot_surface(X, Y, volatilities, cmap='viridis')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Temps')
        ax.set_ylabel('Simulation')
        ax.set_zlabel('Volatilité')
        
        fig.colorbar(surf)
        
        st.pyplot(fig)
        plt.close(fig)
    
    @staticmethod
    def plot_historical_data(prices: pd.Series, returns: pd.Series):
        """
        Affiche les données historiques et les rendements
        
        Args:
            prices: Série temporelle des prix
            returns: Série temporelle des rendements
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Prix historiques
        ax1.plot(prices.index, prices.values, color='blue')
        ax1.set_title("Prix Historiques", fontsize=12)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Prix")
        ax1.grid(True)
        
        # Rendements
        ax2.plot(returns.index, returns.values, color='green')
        ax2.set_title("Rendements", fontsize=12)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Rendement")
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_diagnostics(returns, model_returns, vol_series, model_name="BNS"):
        """
        Génère des graphiques de diagnostic pour le modèle 
        
        Args:
            returns: Série temporelle des rendements réels
            model_returns: Série temporelle des rendements simulés par le modèle
            vol_series: Série temporelle de la volatilité simulée
            model_name: Nom du modèle à afficher dans les titres (défaut: "BNS")
        
        Returns:
            matplotlib.figure.Figure: Figure avec les graphiques de diagnostic
        """
        # Nettoyage des données - supprimer les valeurs NaN et Inf
        def clean_data(data):
            if isinstance(data, pd.Series):
                return data[np.isfinite(data)]
            else:
                return pd.Series([x for x in data if np.isfinite(x)])
                
        # Nettoyer toutes les séries
        returns = clean_data(returns)
        model_returns = clean_data(model_returns)
        vol_series = clean_data(vol_series)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Vérification que nous avons assez de données
        min_data_length = 2  # Minimum requis pour l'autocorrélation
        has_enough_data = len(model_returns) > min_data_length
        
        # Graphique 1: Comparaison des distributions de rendements
        if len(returns) > 0 and len(model_returns) > 0:
            try:
                # Déterminer des limites raisonnables pour les histogrammes
                min_val = max(min(returns.min(), model_returns.min()), -5)
                max_val = min(max(returns.max(), model_returns.max()), 5)
                
                bins = np.linspace(min_val, max_val, min(30, len(model_returns)))
                
                axes[0, 0].hist(returns, bins=bins, alpha=0.5, label='Données réelles', density=True)
                axes[0, 0].hist(model_returns, bins=bins, alpha=0.5, label=f'Modèle {model_name}', density=True)
                axes[0, 0].set_title('Distribution des rendements')
                axes[0, 0].legend()
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f"Erreur d'histogramme: {str(e)}", 
                            ha='center', va='center', transform=axes[0, 0].transAxes)
        else:
            axes[0, 0].text(0.5, 0.5, "Données insuffisantes pour l'histogramme", 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].grid(True)
        
        # Graphique 2: Q-Q Plot
        if len(returns) > min_data_length:
            try:
                from scipy import stats
                # Limiter aux valeurs entre -5 et 5 pour le Q-Q plot
                returns_for_qq = returns[(returns >= -5) & (returns <= 5)]
                if len(returns_for_qq) > 5:
                    stats.probplot(returns_for_qq, dist="norm", plot=axes[0, 1])
                    axes[0, 1].set_title('Q-Q Plot (Normalité des rendements réels)')
                else:
                    axes[0, 1].text(0.5, 0.5, "Données insuffisantes après filtrage des extrêmes", 
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f"Erreur de Q-Q plot: {str(e)}", 
                            ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, "Données insuffisantes pour le Q-Q Plot", 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].grid(True)
        
        # Graphique 3: Volatilité au cours du temps
        if len(vol_series) > 0:
            try:
                # Limiter les valeurs extrêmes pour l'affichage
                vol_for_plot = np.clip(vol_series.values, 0, 2)
                axes[1, 0].plot(vol_for_plot)
                axes[1, 0].set_title(f'Volatilité simulée ({model_name})')
                axes[1, 0].set_xlabel('Temps')
                axes[1, 0].set_ylabel('Volatilité')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f"Erreur de tracé de volatilité: {str(e)}", 
                            ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, "Données de volatilité insuffisantes", 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].grid(True)
        
        # Graphique 4: Autocorrélation des rendements au carré
        if has_enough_data and len(model_returns) >= 5:  # Au moins 5 points pour une ACF significative
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                # Limiter les valeurs extrêmes pour l'ACF
                returns_squared = np.clip(model_returns**2, 0, 25)
                plot_acf(returns_squared, lags=min(20, len(model_returns) // 2), ax=axes[1, 1])
                axes[1, 1].set_title(f'Autocorrélation des rendements au carré ({model_name})')
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f"Erreur lors du calcul de l'autocorrélation: {str(e)}", 
                            ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, "Données insuffisantes pour l'autocorrélation\n(minimum 5 points nécessaires)", 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig

def plot_predictions(ax, price_paths: List[List[float]], bs_prices: List[float]):
    """Plot price predictions for both models.
    
    Args:
        ax: Matplotlib axis to plot on
        price_paths: List of IG-OU price paths
        bs_prices: List of Black-Scholes prices
    """
    # Convert lists to numpy arrays for calculations
    price_paths = np.array(price_paths)
    bs_prices = np.array(bs_prices)
    
    # Calculate statistics for IG-OU paths
    mean_path = np.mean(price_paths, axis=0)
    std_path = np.std(price_paths, axis=0)
    upper_bound = mean_path + 2 * std_path
    lower_bound = mean_path - 2 * std_path
    
    # Plot IG-OU predictions
    time_points = np.arange(len(mean_path))
    ax.plot(time_points, mean_path, 'b-', label='IG-OU Mean Path')
    ax.fill_between(time_points, lower_bound, upper_bound, color='b', alpha=0.2, label='IG-OU 95% CI')
    
    # Plot Black-Scholes prediction
    ax.plot(time_points, bs_prices, 'r--', label='Black-Scholes Path')
    
    # Customize plot
    ax.set_title('Price Predictions')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()

def plot_volatility(ax, vol_paths: List[List[float]]):
    """Plot volatility paths from IG-OU model.
    
    Args:
        ax: Matplotlib axis to plot on
        vol_paths: List of volatility paths
    """
    # Convert to numpy array
    vol_paths = np.array(vol_paths)
    
    # Calculate statistics
    mean_vol = np.mean(vol_paths, axis=0)
    std_vol = np.std(vol_paths, axis=0)
    upper_bound = mean_vol + 2 * std_vol
    lower_bound = mean_vol - 2 * std_vol
    
    # Plot
    time_points = np.arange(len(mean_vol))
    ax.plot(time_points, mean_vol, 'g-', label='Mean Volatility')
    ax.fill_between(time_points, lower_bound, upper_bound, color='g', alpha=0.2, label='95% CI')
    
    # Customize plot
    ax.set_title('Volatility Paths')
    ax.set_xlabel('Days')
    ax.set_ylabel('Volatility')
    ax.grid(True)
    ax.legend()
