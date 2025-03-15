"""Data loading module for predefined assets and timeframes."""
import pandas as pd
import streamlit as st
from pathlib import Path

VALID_ASSETS = ["BTC", "GLE.PA"]
VALID_TIMEFRAMES = ["minute", "hour", "day", "week", "month"]

def load_asset_data(asset: str, timeframe: str) -> pd.DataFrame:
    """Load data from predefined CSV files.
    
    Args:
        asset (str): Asset name ("BTC" or "GLE.PA")
        timeframe (str): Time frame ("minute", "hour", "day", "week", "month")
    
    Returns:
        pd.DataFrame: DataFrame with Date index and Close prices
    """
    if asset not in VALID_ASSETS:
        st.error(f"Invalid asset. Please choose from: {VALID_ASSETS}")
        return None
        
    if timeframe not in VALID_TIMEFRAMES:
        st.error(f"Invalid timeframe. Please choose from: {VALID_TIMEFRAMES}")
        return None

    file_path = Path("assets") / timeframe / f"{asset}.csv"
    
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        if 'Close' not in data.columns:
            st.error(f"Column 'Close' missing in {file_path}")
            return None
        return data['Close'].dropna()
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
