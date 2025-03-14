import pandas as pd
import os
from typing import Optional, Set
import yfinance as yf

class AssetLoader:
    """Chargeur de données avec validation structurelle"""
    
    VALID_ASSETS: Set[str] = {'BTC-USD', 'GLE.PA'}
    VALID_TIMEFRAMES: Set[str] = {'minute', 'hour', 'day', 'week', 'month'}
    
    def load_from_file(self, asset: str, timeframe: str) -> pd.Series:
        """
        Charge les données depuis un fichier local
        
        Args:
            asset: Identifiant de l'actif
            timeframe: Échelle temporelle des données
            
        Returns:
            pd.Series: Série temporelle des prix de clôture
        """
        if asset not in self.VALID_ASSETS:
            raise ValueError(f"Actif non valide: {asset}")
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Échelle temporelle invalide: {timeframe}")
        
        path = f"assets/{timeframe}/{asset}_{timeframe}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier inexistant: {path}")
        
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        self._validate_data(df)
        
        return df['Close'].dropna()
    
    def load_from_yahoo(self, symbol: str, period: str = "1y") -> pd.Series:
        """
        Charge les données depuis Yahoo Finance
        
        Args:
            symbol: Symbole Yahoo Finance
            period: Période de données ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            
        Returns:
            pd.Series: Série temporelle des prix de clôture
        """
        try:
            data = yf.download(symbol, period=period, progress=False)
            self._validate_data(data)
            return data['Close'].dropna()
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement des données Yahoo Finance: {str(e)}")
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Valide la structure et la qualité des données
        
        Args:
            df: DataFrame à valider
            
        Raises:
            ValueError: Si les données ne respectent pas la structure attendue
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Les données doivent être un DataFrame")
            
        if 'Close' not in df.columns:
            raise ValueError("Colonne 'Close' manquante")
            
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("L'index doit être de type DatetimeIndex")
            
        if df['Close'].dtype.kind not in 'iuf':
            raise ValueError("Les prix doivent être numériques")
            
        if df['Close'].isna().all():
            raise ValueError("Aucune donnée valide dans la colonne Close")
            
        # Vérification de l'ordre chronologique
        if not df.index.is_monotonic_increasing:
            raise ValueError("Les données ne sont pas dans l'ordre chronologique")
