import streamlit as st
from typing import Dict, Any, Tuple
import pandas as pd

class Sidebar:
    """Gestion de la barre latérale de l'interface"""
    
    @staticmethod
    def render() -> Dict[str, Any]:
        """
        Affiche et gère les contrôles de la barre latérale
        
        Returns:
            Dict[str, Any]: Configuration sélectionnée par l'utilisateur
        """
        with st.sidebar:
            st.title("Configuration")
            
            # Sélection de la source de données
            data_source = st.selectbox(
                "Source des données",
                ["Yahoo Finance", "Fichier Excel/CSV", "Données d'exemple"]
            )
            
            # Configuration selon la source
            if data_source == "Yahoo Finance":
                config = Sidebar._yahoo_config()
            elif data_source == "Fichier Excel/CSV":
                config = Sidebar._file_config()
            else:
                config = Sidebar._example_config()
            
            # Paramètres de simulation communs
            st.markdown("---")
            st.subheader("Paramètres de simulation")
            
            config.update({
                'n_simulations': st.number_input(
                    "Nombre de simulations",
                    min_value=1,
                    max_value=10000,
                    value=1000
                ),
                'horizon': st.number_input(
                    "Horizon de prédiction (jours)",
                    min_value=1,
                    max_value=252,
                    value=30
                ),
                'model': st.selectbox(
                    "Modèle",
                    ["IG-OU", "Black-Scholes", "Les deux"]
                )
            })
            
            return config
    
    @staticmethod
    def _yahoo_config() -> Dict[str, Any]:
        """Configuration pour Yahoo Finance"""
        return {
            'source': 'yahoo',
            'symbol': st.text_input("Symbole Yahoo Finance", "BTC-USD"),
            'period': st.selectbox(
                "Période historique",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3
            )
        }
    
    @staticmethod
    def _file_config() -> Dict[str, Any]:
        """Configuration pour le chargement de fichier"""
        uploaded_file = st.file_uploader("Fichier de données", type=["csv", "xlsx"])
        
        return {
            'source': 'file',
            'file': uploaded_file,
            'date_column': st.text_input("Colonne date", "Date") if uploaded_file else None,
            'price_column': st.text_input("Colonne prix", "Close") if uploaded_file else None
        }
    
    @staticmethod
    def _example_config() -> Dict[str, Any]:
        """Configuration pour les données d'exemple"""
        return {
            'source': 'example',
            'asset': st.selectbox("Actif", ["BTC-USD", "GLE.PA"]),
            'timeframe': st.selectbox(
                "Échelle temporelle",
                ["minute", "hour", "day", "week", "month"]
            )
        }
