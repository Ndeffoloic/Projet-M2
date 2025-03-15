import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from ui.components.sidebar import render_sidebar
from ui.components.visualizations import plot_predictions, plot_volatility
from ui.helpers import show_statistics

@pytest.fixture
def mock_streamlit():
    """Create a mock for streamlit."""
    with patch('streamlit.sidebar') as mock_sidebar:
        with patch('streamlit.selectbox') as mock_selectbox:
            with patch('streamlit.number_input') as mock_number_input:
                yield {
                    'sidebar': mock_sidebar,
                    'selectbox': mock_selectbox,
                    'number_input': mock_number_input
                }

def test_app_title():
    """Test application title."""
    with patch('streamlit.title') as mock_title:
        from app import main
        main()
        mock_title.assert_called_once_with("Prediction de Prix des Actifs avec Modele IG-OU")

def test_data_source_selector():
    """Test data source selection."""
    with patch('streamlit.sidebar.selectbox') as mock_selectbox:
        mock_selectbox.side_effect = ["BTC-USD", "day"]
        config = render_sidebar()
        assert config["asset"] == "BTC-USD"
        assert config["timeframe"] == "day"

def test_parameter_inputs():
    """Test parameter input handling."""
    with patch('streamlit.sidebar.number_input') as mock_number_input:
        mock_number_input.side_effect = [100, 0.01, 1.0]
        config = render_sidebar()
        assert config["n_simulations"] == 100
        assert config["a"] == 0.01
        assert config["b"] == 1.0

def test_error_handling():
    """Test error display."""
    with patch('streamlit.error') as mock_error:
        from core.data.loader import load_asset_data
        load_asset_data("INVALID", "day")
        mock_error.assert_called_once()

@pytest.fixture
def sample_data():
    """Create sample data for visualization tests."""
    n_simulations = 10
    days = 30
    
    price_paths = np.random.normal(100, 10, (n_simulations, days))
    vol_paths = np.abs(np.random.normal(0.2, 0.05, (n_simulations, days)))
    bs_prices = np.random.normal(100, 10, days)
    
    return price_paths, vol_paths, bs_prices

def test_plot_predictions(sample_data):
    """Test prediction plotting function."""
    price_paths, _, bs_prices = sample_data
    
    # Create mock axis
    mock_ax = MagicMock()
    
    # Call plotting function
    plot_predictions(mock_ax, price_paths, bs_prices)
    
    # Verify plot calls
    assert mock_ax.plot.call_count >= 2  # At least mean path and BS path
    assert mock_ax.set_title.called
    assert mock_ax.legend.called
    assert mock_ax.grid.called

def test_plot_volatility(sample_data):
    """Test volatility plotting function."""
    _, vol_paths, _ = sample_data
    
    # Create mock axis
    mock_ax = MagicMock()
    
    # Call plotting function
    plot_volatility(mock_ax, vol_paths)
    
    # Verify plot calls
    assert mock_ax.plot.call_count >= 1
    assert mock_ax.set_title.called
    assert mock_ax.legend.called
    assert mock_ax.grid.called

def test_show_statistics(sample_data):
    """Test statistics display function."""
    price_paths, vol_paths, bs_prices = sample_data
    
    with patch('streamlit.dataframe') as mock_dataframe:
        with patch('streamlit.write') as mock_write:
            # Call statistics function
            show_statistics(price_paths, vol_paths, bs_prices)
            
            # Verify display calls
            assert mock_dataframe.called
            assert mock_write.called
