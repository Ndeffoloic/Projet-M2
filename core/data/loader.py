"""Data loading module for predefined assets and timeframes."""
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import streamlit as st

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("finance_loader")

VALID_ASSETS = ["BTC-USD", "GLE.PA", "NVIDIA"]
VALID_TIMEFRAMES = ["minute", "hour", "day", "week", "month"]

# Limites raisonnables pour les valeurs financières (à ajuster selon vos actifs)
# Ces limites dépendent de la nature de vos actifs (crypto, actions, etc.)
MIN_PRICE = 0.001  # Prix minimal positif 
MAX_PRICE = 100000.0  # Prix maximal raisonnable
MAX_DAILY_RETURN = 0.5  # 50% max de variation quotidienne (très conservateur pour crypto)

def load_asset_data(asset: str, timeframe: str) -> pd.Series:
    """Charge les données d'un actif financier à partir de fichiers CSV prédéfinis.
    
    Cette fonction réalise plusieurs validations et nettoyages :
    - Vérifie que l'actif et le timeframe sont valides
    - Nettoie les données de prix (supprime quotes, virgules)
    - Filtre les valeurs aberrantes (NaN, Inf, négatifs, outliers)
    - Assure la continuité temporelle
    
    Args:
        asset (str): Nom de l'actif ("BTC-USD", "GLE.PA" ou "NVIDIA")
        timeframe (str): Échelle temporelle ("minute", "hour", "day", "week", "month")
    
    Returns:
        pd.Series: Series avec index Date et valeurs de prix nettoyées
              ou None en cas d'erreur
    """
    # --- 1. Validation des paramètres d'entrée ---
    if asset not in VALID_ASSETS:
        st.error(f"Actif invalide. Veuillez choisir parmi: {VALID_ASSETS}")
        return None
        
    if timeframe not in VALID_TIMEFRAMES:
        st.error(f"Timeframe invalide. Veuillez choisir parmi: {VALID_TIMEFRAMES}")
        return None

    file_path = Path("assets") / timeframe / f"{asset}_{timeframe}.csv"
    
    try:
        # --- 2. Chargement initial du fichier ---
        logger.info(f"Chargement des données pour {asset} ({timeframe}) depuis {file_path}")
        data = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            index_col='Date',
            dtype={'Close': str, 'Price': str}  # Lire comme string d'abord pour gérer les quotes et virgules
        )
        
        # --- 3. Identification de la colonne de prix ---
        if 'Close' in data.columns:
            price_column = 'Close'
        elif 'Price' in data.columns:
            price_column = 'Price'
        else:
            st.error(f"Aucune colonne 'Close' ou 'Price' trouvée dans {file_path}")
            return None
            
        # --- 4. Nettoyage et conversion de la colonne de prix ---
        logger.info(f"Nettoyage de la colonne {price_column}")
        # Supprimer quotes, virgules, puis convertir en float
        price_series = (data[price_column]
                      .str.replace('"', '', regex=False)
                      .str.replace(',', '', regex=False)
                      .astype(float))
        
        # --- 5. Filtrage des valeurs aberrantes ---
        # 5.1 Suppression des NaN
        initial_length = len(price_series)
        price_series = price_series.dropna()
        nan_count = initial_length - len(price_series)
        if nan_count > 0:
            logger.warning(f"{nan_count} valeurs NaN supprimées")
            
        # 5.2 Filtrage des valeurs négatives ou nulles (normalement impossibles pour des prix)
        valid_prices = price_series > 0
        if not valid_prices.all():
            invalid_count = (~valid_prices).sum()
            logger.warning(f"{invalid_count} prix négatifs ou nuls détectés et filtrés")
            price_series = price_series[valid_prices]
            
        # 5.3 Filtrage des valeurs infinies
        finite_prices = np.isfinite(price_series)
        if not finite_prices.all():
            inf_count = (~finite_prices).sum()
            logger.warning(f"{inf_count} valeurs infinies détectées et filtrées")
            price_series = price_series[finite_prices]
            
        # 5.4 Filtrage des outliers extrêmes (basé sur des limites raisonnables)
        reasonable_prices = (price_series >= MIN_PRICE) & (price_series <= MAX_PRICE)
        if not reasonable_prices.all():
            outlier_count = (~reasonable_prices).sum()
            logger.warning(f"{outlier_count} outliers extrêmes détectés et filtrés")
            price_series = price_series[reasonable_prices]
        
        # 5.5 Détection des sauts anormaux (variations quotidiennes trop importantes)
        if len(price_series) > 1:
            returns = price_series.pct_change().abs()
            extreme_returns = returns > MAX_DAILY_RETURN
            extreme_returns.iloc[0] = False  # Ignorer la première valeur (NaN)
            
            if extreme_returns.any():
                extreme_count = extreme_returns.sum()
                logger.warning(f"{extreme_count} variations anormales détectées (>{MAX_DAILY_RETURN*100}%)")
                # Option 1: juste un avertissement (conserver les données)
                # Option 2: filtrer ces points (décommentez la ligne ci-dessous)
                # price_series = price_series[~extreme_returns] 
        
        # --- 6. Vérification finale ---
        if len(price_series) == 0:
            st.error("Aucune donnée valide trouvée après filtrage")
            return None
        
        filtered_count = initial_length - len(price_series)
        if filtered_count > 0:
            percentage = round((filtered_count / initial_length) * 100, 2)
            if percentage > 10:  # Alerte si plus de 10% des données filtrées
                st.warning(f"{filtered_count} points de données filtrés ({percentage}% des données d'origine)")
            else:
                logger.info(f"{filtered_count} points de données filtrés ({percentage}% des données d'origine)")
            
        # --- 7. Retour des données nettoyées ---
        logger.info(f"Données chargées avec succès: {len(price_series)} points")
        # Retourner l'ensemble complet des données sans troncation
        return price_series
        
    except FileNotFoundError:
        st.error(f"Fichier non trouvé: {file_path}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        # Pour le débogage
        st.error(f"Chemin du fichier: {file_path}")
        import traceback
        st.error(traceback.format_exc())
        return None