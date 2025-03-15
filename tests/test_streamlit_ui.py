import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest
import yfinance as yf
from WCE2009_streamlit import main

@pytest.fixture
def sample_stock_data():
    """Fixture pour générer des données de test"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.uniform(1000000, 2000000, len(dates))
    }, index=dates)
    return data

@pytest.fixture
def mock_yf_download(sample_stock_data):
    """Mock pour yfinance.download"""
    with patch('yfinance.download') as mock:
        mock.return_value = sample_stock_data
        yield mock

def test_app_title():
    """Test le titre de l'application"""
    at = AppTest.from_function(main)
    at.run()
    assert "Prédiction de Prix et Volatilité" in at.title

def test_data_source_selector():
    """Test le sélecteur de source de données"""
    at = AppTest.from_function(main)
    at.run()
    
    # Vérifier que le sélecteur existe et contient les bonnes options
    data_source = at.selectbox("Source des données")
    assert data_source is not None
    assert "Yahoo Finance" in data_source.options
    assert "Fichier Excel/CSV" in data_source.options
    assert "Données d'exemple" in data_source.options

@pytest.mark.usefixtures("mock_yf_download")
def test_yahoo_finance_download():
    """Test le téléchargement de données Yahoo Finance"""
    at = AppTest.from_function(main)
    
    # Simuler la sélection de Yahoo Finance
    at.selectbox("Source des données").select("Yahoo Finance")
    
    # Remplir le formulaire
    with at.form("yahoo_form"):
        at.text_input("Symbole Yahoo Finance (ex: AAPL)").input("AAPL")
        at.selectbox("Période").select("1mo")
        at.form_submit_button().click()
    
    # Vérifier que le graphique est affiché
    assert len(at.line_chart) > 0

def test_file_upload():
    """Test le téléchargement de fichier"""
    at = AppTest.from_function(main)
    
    # Simuler la sélection de Fichier Excel/CSV
    at.selectbox("Source des données").select("Fichier Excel/CSV")
    
    # Créer un DataFrame de test
    test_data = pd.DataFrame({
        'Close': [100, 101, 102, 103]
    })
    
    # Simuler le téléchargement d'un fichier CSV
    csv_content = test_data.to_csv()
    at.file_uploader("Télécharger fichier").upload(csv_content.encode(), "test.csv")
    
    # Vérifier que le graphique est affiché
    assert len(at.line_chart) > 0

def test_parameter_inputs():
    """Test les entrées de paramètres"""
    at = AppTest.from_function(main)
    at.run()
    
    # Vérifier que les widgets de paramètres existent
    assert at.number_input("Nombre de simulations") is not None
    assert at.number_input("Jours de prédiction") is not None

def test_error_handling():
    """Test la gestion des erreurs"""
    at = AppTest.from_function(main)
    
    # Simuler une erreur de téléchargement Yahoo Finance
    with patch('yfinance.download', side_effect=Exception("Test error")):
        at.selectbox("Source des données").select("Yahoo Finance")
        with at.form("yahoo_form"):
            at.text_input("Symbole Yahoo Finance (ex: AAPL)").input("INVALID")
            at.selectbox("Période").select("1mo")
            at.form_submit_button().click()
        
        # Vérifier que l'erreur est affichée
        assert "error" in at.error[0].value.lower()

"""Test suite for Streamlit UI components."""
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

def test_sidebar_rendering(mock_streamlit):
    """Test sidebar component rendering."""
    # Mock return values
    mock_streamlit['selectbox'].side_effect = ['BTC', 'minute']
    mock_streamlit['number_input'].side_effect = [100, 2.2395e-7, 1.0]
    
    # Call sidebar render
    config = render_sidebar()
    
    # Check returned configuration
    assert isinstance(config, dict)
    assert 'asset' in config
    assert 'timeframe' in config
    assert 'n_simulations' in config
    assert 'a' in config
    assert 'b' in config

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
