"""Data loading module for predefined assets and timeframes."""
from pathlib import Path

import pandas as pd
import streamlit as st

VALID_ASSETS = ["BTC-USD", "GLE.PA", "NVIDIA"]
VALID_TIMEFRAMES = ["minute", "hour", "day", "week", "month"]

def load_asset_data(asset: str, timeframe: str) -> pd.Series:
    """Load data from predefined CSV files.
    
    Args:
        asset (str): Asset name ("BTC-USD", "GLE.PA" or "NVIDIA")
        timeframe (str): Time frame ("minute", "hour", "day", "week", "month")
    
    Returns:
        pd.Series: Series with Date index and Close/Price values
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
            dtype={'Close': str, 'Price': str}  # Read as string first to handle quotes and commas
        )
        
        # Try to find 'Close' column, if not found, try 'Price' column
        if 'Close' in data.columns:
            price_column = 'Close'
        elif 'Price' in data.columns:
            price_column = 'Price'
        else:
            st.error(f"Neither 'Close' nor 'Price' column found in {file_path}")
            return None
            
        # Clean the price column:
        # 1. Remove quotes
        # 2. Remove commas (thousand separators)
        # 3. Convert to float
        price_series = (data[price_column]
                      .str.replace('"', '', regex=False)
                      .str.replace(',', '', regex=False)
                      .astype(float))
        
        # Remove any NaN values
        price_series = price_series.dropna()
        
        if len(price_series) == 0:
            st.error("No valid numeric data found in price column")
            return None
            
        # Return a Series with Date index (not a DataFrame)
        return price_series
        
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