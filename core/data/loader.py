"""Data loading module for predefined assets and timeframes."""
from pathlib import Path

import pandas as pd
import streamlit as st

VALID_ASSETS = ["BTC-USD", "GLE.PA"]
VALID_TIMEFRAMES = ["minute", "hour", "day", "week", "month"]

def load_asset_data(asset: str, timeframe: str) -> pd.DataFrame:
    """Load data from predefined CSV files.
    
    Args:
        asset (str): Asset name ("BTC-USD" or "GLE.PA")
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

    file_path = Path("assets") / timeframe / f"{asset}_{timeframe}.csv"
    
    try:
        # Read CSV without immediate conversion to numeric values
        data = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            index_col='Date',
            dtype={'Close': str}  # Read as string first to handle quotes and commas
        )
        
        if 'Close' not in data.columns:
            st.error(f"Column 'Close' missing in {file_path}")
            return None
            
        # Clean the Close column:
        # 1. Remove quotes
        # 2. Remove commas (thousand separators)
        # 3. Convert to float
        close_series = (data['Close']
                      .str.replace('"', '', regex=False)
                      .str.replace(',', '', regex=False)
                      .astype(float))
        
        # Remove any NaN values
        close_series = close_series.dropna()
        
        if len(close_series) == 0:
            st.error("No valid numeric data found in Close column")
            return None
            
        # Retourner une DataFrame avec l'index Date et la colonne Close
        result_df = pd.DataFrame({'Close': close_series}, index=close_series.index)
        return result_df
        
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # For debugging
        st.error(f"File path: {file_path}")
        import traceback
        st.error(traceback.format_exc())
        return None