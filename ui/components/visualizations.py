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
