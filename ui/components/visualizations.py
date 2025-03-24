from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


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
            if data is None:
                return None
            try:
                # Conversion en array numpy et aplatissement
                np_array = np.array(data, dtype=float)
                flattened = np_array.ravel()  # Convertit en 1D quel que soit l'input
                finite_mask = np.isfinite(flattened)
                
                if not np.any(finite_mask):
                    return None
                    
                cleaned = pd.Series(flattened[finite_mask])
                return cleaned if not cleaned.empty else None
                
            except Exception as e:
                print(f"Erreur de nettoyage: {str(e)}")
                return None
                
        # Nettoyer toutes les séries
        returns = clean_data(returns)
        model_returns = clean_data(model_returns)
        
        # Ne nettoyer vol_series que si elle n'est pas None
        if vol_series is not None:
            vol_series = clean_data(vol_series)
        
        # Utiliser une figure plus grande pour les graphiques
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
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
                
                axes[0].hist(returns, bins=bins, alpha=0.5, label='Données réelles', density=True)
                axes[0].hist(model_returns, bins=bins, alpha=0.5, label=f'Modèle {model_name}', density=True)
                axes[0].set_title('Distribution des rendements', fontsize=14)
                axes[0].legend(fontsize=12)
            except Exception as e:
                axes[0].text(0.5, 0.5, f"Erreur d'histogramme: {str(e)}", 
                            ha='center', va='center', transform=axes[0].transAxes)
        else:
            axes[0].text(0.5, 0.5, "Données insuffisantes pour l'histogramme", 
                        ha='center', va='center', transform=axes[0].transAxes)
        axes[0].grid(True)
        
        # Graphique 2: Q-Q Plot
        if len(returns) > min_data_length:
            try:
                from scipy import stats

                # Utiliser l'ensemble des données pour le Q-Q plot
                stats.probplot(returns, dist="norm", plot=axes[1])
                axes[1].set_title('Q-Q Plot (Normalité des rendements réels)', fontsize=14)
            except Exception as e:
                axes[1].text(0.5, 0.5, f"Erreur de Q-Q plot: {str(e)}", 
                            ha='center', va='center', transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, "Données insuffisantes pour le Q-Q Plot", 
                        ha='center', va='center', transform=axes[1].transAxes)
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_autocorrelation_comparison(actual_returns, igou_returns, bs_returns, bns_returns, lags=40):
        """Figure 2: True vs Estimated Autocorrelation
        
        Args:
            actual_returns: Rendements réels observés
            igou_returns: Rendements simulés par le modèle IG-OU
            bs_returns: Rendements simulés par le modèle Black-Scholes
            bns_returns: Rendements simulés par le modèle BNS
            lags: Nombre de retards pour l'autocorrélation
            
        Returns:
            matplotlib.figure.Figure: Figure avec les autocorrélations comparées
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # S'assurer que les séries sont des pd.Series
        def ensure_series(data):
            if not isinstance(data, pd.Series):
                return pd.Series(data)
            return data
            
        actual_returns = ensure_series(actual_returns)
        igou_returns = ensure_series(igou_returns)
        bs_returns = ensure_series(bs_returns)
        bns_returns = ensure_series(bns_returns)
        
        # Limiter lags à la moitié de la longueur des données minimale
        min_length = min(len(actual_returns), len(igou_returns), len(bs_returns), len(bns_returns))
        max_lags = min(lags, min_length // 2)
        
        # Calcul des autocorrélations manuellement pour pouvoir les superposer
        from statsmodels.tsa.stattools import acf

        # Calculer l'ACF pour chaque série
        try:
            acf_actual = acf(actual_returns, nlags=max_lags, fft=True)
            acf_igou = acf(igou_returns, nlags=max_lags, fft=True)
            acf_bs = acf(bs_returns, nlags=max_lags, fft=True)
            acf_bns = acf(bns_returns, nlags=max_lags, fft=True)
            
            # Créer l'axe des x
            lags_x = range(len(acf_actual))
            
            # Tracer les autocorrélations
            ax.stem(lags_x, acf_actual, markerfmt='bo', linefmt='b-', basefmt='b-', label='Données réelles')
            ax.plot(lags_x, acf_igou, 'r--', label='IG-OU', alpha=0.7)
            ax.plot(lags_x, acf_bs, 'g-.', label='Black-Scholes', alpha=0.7)
            ax.plot(lags_x, acf_bns, 'm:', label='BNS', alpha=0.7, linewidth=2)
            
            ax.set_title("Autocorrélation réelle vs estimée ", fontsize=16)
            ax.set_xlabel("Retards (lags)", fontsize=14)
            ax.set_ylabel("Autocorrélation", fontsize=14)
            ax.grid(True)
            ax.legend(fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f"Erreur lors du calcul des autocorrélations: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_returns_comparison(actual_returns, igou_returns, bs_returns, bns_returns):
        """Figure 6: Historical vs Estimated Returns
        
        Args:
            actual_returns: Rendements réels observés
            igou_returns: Rendements simulés par le modèle IG-OU
            bs_returns: Rendements simulés par le modèle Black-Scholes
            bns_returns: Rendements simulés par le modèle BNS
            
        Returns:
            matplotlib.figure.Figure: Figure avec les rendements comparés
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Convertir en numpy arrays si ce n'est pas déjà le cas
        actual_returns = np.array(actual_returns)
        igou_returns = np.array(igou_returns)
        bs_returns = np.array(bs_returns)
        bns_returns = np.array(bns_returns)
        
        # Déterminer la longueur minimale pour l'affichage
        min_length = min(len(actual_returns), len(igou_returns), len(bs_returns), len(bns_returns))
        
        # Créer les points temporels
        time_points = np.arange(min_length)
        
        # Tracer les rendements
        ax.plot(time_points, actual_returns[:min_length], 'b-', label='Rendements réels', linewidth=1)
        ax.plot(time_points, igou_returns[:min_length], 'r--', label='IG-OU', alpha=0.7)
        ax.plot(time_points, bs_returns[:min_length], 'g-.', label='Black-Scholes', alpha=0.7)
        ax.plot(time_points, bns_returns[:min_length], 'm:', label='BNS', alpha=0.7, linewidth=2)
        
        ax.set_title("Comparaison des rendements historiques et estimés ", fontsize=16)
        ax.set_xlabel("Temps", fontsize=14)
        ax.set_ylabel("Rendements", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_residuals_acf(actual_returns, model_returns, model_name):
        """Figure 8: ACF of Squared Residuals
        
        Args:
            actual_returns: Rendements réels observés
            model_returns: Rendements simulés par le modèle
            model_name: Nom du modèle à afficher
            
        Returns:
            matplotlib.figure.Figure: Figure avec l'ACF des résidus carrés
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Convertir en séries si nécessaire
        if not isinstance(actual_returns, pd.Series):
            actual_returns = pd.Series(actual_returns)
        if not isinstance(model_returns, pd.Series):
            model_returns = pd.Series(model_returns)
            
        try:
            # Réinitialiser les index pour éviter les problèmes de comparaison
            actual_returns = actual_returns.reset_index(drop=True)
            model_returns = model_returns.reset_index(drop=True)
            
            # Limiter model_returns à la longueur de actual_returns
            min_length = min(len(actual_returns), len(model_returns))
            
            # Calculer les résidus en utilisant les valeurs numpy
            residuals = actual_returns.values[:min_length] - model_returns.values[:min_length]
            squared_residuals = residuals ** 2
            
            # Tracer l'ACF
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(squared_residuals, lags=min(40, min_length // 2), ax=ax)
            
            ax.set_title(f'ACF des résidus carrés ({model_name}) ', fontsize=16)
            ax.set_xlabel("Retards (lags)", fontsize=14)
            ax.set_ylabel("Autocorrélation", fontsize=14)
            ax.grid(True)
        except Exception as e:
            ax.text(0.5, 0.5, f"Erreur lors du calcul de l'ACF des résidus: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
        
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
