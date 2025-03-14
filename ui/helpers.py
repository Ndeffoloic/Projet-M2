import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def display_error(message: str) -> None:
    """Affiche un message d'erreur"""
    st.error(f"🚨 {message}")

def display_success(message: str) -> None:
    """Affiche un message de succès"""
    st.success(f"✅ {message}")

def display_info(message: str) -> None:
    """Affiche un message d'information"""
    st.info(f"ℹ️ {message}")

def display_parameters(params: dict) -> None:
    """
    Affiche les paramètres estimés
    
    Args:
        params: Dictionnaire des paramètres
    """
    st.subheader("Paramètres estimés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for key, value in params.items():
            st.write(f"{key}: {value:.6f}")
    
    with col2:
        st.markdown("""
        **Interprétation:**
        - μ: Drift (tendance)
        - σ²: Variance de la volatilité
        - λ: Vitesse de retour à la moyenne
        """)

def format_data_summary(data: pd.Series) -> None:
    """
    Affiche un résumé des données
    
    Args:
        data: Série temporelle à analyser
    """
    st.subheader("Résumé des données")
    
    summary = pd.DataFrame({
        'Statistique': [
            'Nombre d\'observations',
            'Première date',
            'Dernière date',
            'Prix minimum',
            'Prix maximum',
            'Prix moyen',
            'Écart-type',
            'Skewness',
            'Kurtosis'
        ],
        'Valeur': [
            len(data),
            data.index.min().strftime('%Y-%m-%d'),
            data.index.max().strftime('%Y-%m-%d'),
            f"{data.min():.2f}",
            f"{data.max():.2f}",
            f"{data.mean():.2f}",
            f"{data.std():.2f}",
            f"{data.skew():.2f}",
            f"{data.kurtosis():.2f}"
        ]
    })
    
    st.dataframe(summary, hide_index=True)

def check_data_quality(data: pd.Series) -> Tuple[bool, list]:
    """
    Vérifie la qualité des données
    
    Args:
        data: Série temporelle à vérifier
        
    Returns:
        Tuple[bool, list]: (données valides?, liste des problèmes)
    """
    issues = []
    
    # Vérifier les valeurs manquantes
    if data.isna().any():
        issues.append(f"Données manquantes: {data.isna().sum()} valeurs")
    
    # Vérifier les valeurs négatives
    if (data < 0).any():
        issues.append("Présence de prix négatifs")
    
    # Vérifier les valeurs extrêmes
    z_scores = np.abs((data - data.mean()) / data.std())
    if (z_scores > 5).any():
        n_outliers = (z_scores > 5).sum()
        issues.append(f"Valeurs extrêmes détectées: {n_outliers} observations")
    
    # Vérifier la fréquence des données
    time_diffs = data.index.to_series().diff()
    if time_diffs.nunique() > 1:
        issues.append("Fréquence d'échantillonnage irrégulière")
    
    return len(issues) == 0, issues
