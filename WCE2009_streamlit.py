import time
from datetime import datetime, timedelta
from io import StringIO
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import (norm, invgauss, shapiro, jarque_bera)
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
import seaborn as sns
import statsmodels.api as sm

# Importer les fonctions nécessaires
from scipy.stats import invgauss

def generate_ig(a, b, size):
    """Génère des variables aléatoires Inverse Gaussiennes (Algorithme du document)"""
    # Éviter les problèmes si a ou b sont très proches de 0
    a = max(a, 1e-10)
    b = max(b, 1e-10)
    
    N = np.random.normal(0, 1, size)
    Y = N**2
    X1 = (a/b) + Y/(2*b**2) - np.sqrt(4*a*b*Y + Y**2)/(2*b**2)
    
    # Éviter les valeurs nulles pour X1
    X1 = np.maximum(X1, 1e-10)
    
    U = np.random.uniform(0, 1, size)
    mask = U <= a/(a + X1*b)
    
    # Calculer la seconde valeur possible en gérant les divisions par zéro
    second_value = np.zeros_like(X1)
    np.divide(a**2, b**2 * X1, out=second_value, where=X1!=0)
    
    # Si X1=0, utiliser une valeur très grande mais finie
    second_value[X1 == 0] = 1e10
    
    # Sélectionner X1 ou second_value selon le masque
    X = np.where(mask, X1, second_value)
    return X

def simulate_ig_ou(X0, lambda_, a, b, T=30, dt=1/252):
    """Simule le processus IG-OU pour la volatilité (Formule 3.17)"""
    n_steps = int(T/dt)
    X = np.zeros(n_steps)
    X[0] = X0
    
    for t in range(1, n_steps):
        h = dt
        L = generate_ig(a*h, b, 1)[0]
        X[t] = np.exp(-lambda_*h)*X[t-1] + L
    
    return X[:T]  # Retourner exactement T valeurs

# Nouvelle fonction pour Black-Scholes
def simulate_bs(S0, mu, sigma, days=30):
    """Simule les prix avec un modèle de Black-Scholes"""
    dt = 1/252
    prices = [S0]
    for _ in range(days-1):  # -1 car on a déjà S0
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal()
        prices.append(prices[-1] * np.exp(drift + shock))
    return np.array(prices)

def simulate_bns(S0, lambda_, a, b, theta, T=30, dt=1/252):
    """Simule le modèle Barndorff-Nielsen & Shephard"""
    n_steps = int(T/dt)
    X = np.zeros(n_steps)
    prices = np.zeros(n_steps)
    X[0] = S0
    prices[0] = S0
    
    for t in range(1, n_steps):
        # Processus de volatilité
        dL = generate_ig(a*dt, b, 1)[0]
        X[t] = X[t-1] * np.exp(-lambda_*dt) + dL
        
        # Processus de prix
        drift = (theta - 0.5 * X[t]) * dt
        shock = np.sqrt(X[t]) * np.sqrt(dt) * np.random.normal()
        prices[t] = prices[t-1] * np.exp(drift + shock)
    
    return prices[:T]

def estimate_parameters(returns):
    """Estime μ, σ² et λ selon les formules du document (Section 3)"""
    # Vérification que returns n'est pas vide ou contient des NaN
    returns = returns.dropna()
    if len(returns) <= 1:
        return 0.0001, 0.01, 0.1  # Valeurs par défaut
    
    mu = returns.mean()
    sigma_sq = 2 * returns.var()
    
    # Calcul sécurisé de l'autocorrélation lag-1
    if len(returns) > 1:
        try:
            # Utiliser pandas pour l'autocorrélation, plus robuste contre les NaN
            rho1 = returns.autocorr(lag=1)
            # Vérifier si rho1 est NaN
            if pd.isna(rho1):
                rho1 = 0.1  # Valeur par défaut
        except Exception:
            rho1 = 0.1
    
    # Éviter un lambda négatif ou zéro
    if rho1 >= 0.999:
        rho1 = 0.999
    elif rho1 <= 0:
        rho1 = 0.001
        
    lambda_ = -np.log(max(rho1, 1e-6))
    
    return mu, sigma_sq, lambda_

# Modified function to perform backtesting of prediction models
def perform_backtesting(data, test_percentage=0.3, n_simulations=100):
    """
    Performs backtesting of prediction models by using historical data to predict the last test_percentage of data points.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with 'Price' price column
    test_percentage (float): Percentage of data points to use for testing (default: 0.3 or 30%)
    n_simulations (int): Number of Monte Carlo simulations to run (default: 100)
    
    Returns:
    dict: Dictionary with predictions and metrics for both models
    """
    # Ensure data is sorted by date
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    
    # Remove NaN values from the data before splitting
    data = data.dropna(subset=['Price'])
    
    # Calculate number of test data points based on percentage
    n_test = int(len(data) * test_percentage)
    
    # Check if we have enough data after dropping NaNs
    if len(data) <= n_test:
        st.error(f"Not enough valid data for backtesting. Need more than {n_test} non-NaN data points.")
        return None
    
    # Split data into training and testing sets
    train_data = data[:-n_test].copy()
    test_data = data[-n_test:].copy()
    
    # Calculate returns for the training set
    returns = train_data['Price'].pct_change(fill_method=None).dropna()
    
    # Estimate parameters from training data
    mu, sigma_sq, lambda_ = estimate_parameters(returns)
    
    # Get the last price from training data for initialization
    last_price = train_data['Price'].iloc[-1]
    init_vol = max(min(returns.std(), 0.05), 0.001)
    
    # Initialize arrays for predictions
    igou_paths = np.zeros((n_simulations, n_test))
    bns_paths = np.zeros((n_simulations, n_test))
    
    # Run IG-OU model simulations
    for i in range(n_simulations):
        try:
            # Simulate volatility
            vol = simulate_ig_ou(X0=init_vol, lambda_=max(lambda_, 0.001), a=2.2395e-7, b=1.0, T=n_test)
            
            # Simulate prices with the simulated volatility
            prices = [last_price]
            for t in range(n_test-1):  # Nous avons besoin de n_test-1 itérations car prices commence déjà avec last_price
                drift = mu
                shock = vol[t] * np.random.normal()
                prices.append(prices[-1] * np.exp(drift + shock))
            
            igou_paths[i] = prices
        except Exception as e:
            st.error(f"Error in IG-OU simulation {i+1}: {str(e)}")
            # Fill with the last successful simulation or with zeros
            igou_paths[i] = igou_paths[max(0, i-1)] if i > 0 else np.zeros(n_test)
    
    # Run BNS model simulations
    for i in range(n_simulations):
        try:
            bns_prices = simulate_bns(last_price, lambda_, a=2.2395e-7, b=1.0, theta=0.1, T=n_test)
            bns_paths[i] = bns_prices
        except Exception as e:
            st.error(f"Error in BNS simulation {i+1}: {str(e)}")
            # Fill with the last successful simulation or with zeros
            bns_paths[i] = bns_paths[max(0, i-1)] if i > 0 else np.zeros(n_test)
    
    # Calculate mean prediction for IG-OU model
    igou_mean_prediction = np.mean(igou_paths, axis=0)
    
    # Run Black-Scholes simulation
    bs_prediction = simulate_bs(last_price, mu, returns.std(), days=n_test)
    
    # Get actual test values
    actual_prices = test_data['Price'].values
    
    # Ensure there are no NaN values in any of the arrays
    valid_indices = ~np.isnan(actual_prices) & ~np.isnan(igou_mean_prediction) & ~np.isnan(bs_prediction)
    
    if np.any(valid_indices):
        # Filter out NaN values
        filtered_actual = actual_prices[valid_indices]
        filtered_igou = igou_mean_prediction[valid_indices]
        filtered_bs = bs_prediction[valid_indices]
        filtered_bns = np.mean(bns_paths, axis=0)[valid_indices]
        
        # Calculate metrics
        igou_rmse = sqrt(mean_squared_error(filtered_actual, filtered_igou))
        bs_rmse = sqrt(mean_squared_error(filtered_actual, filtered_bs))
        bns_rmse = sqrt(mean_squared_error(filtered_actual, filtered_bns))
        
        igou_mae = mean_absolute_error(filtered_actual, filtered_igou)
        bs_mae = mean_absolute_error(filtered_actual, filtered_bs)
        bns_mae = mean_absolute_error(filtered_actual, filtered_bns)
        
        # Mean Absolute Percentage Error (MAPE)
        igou_mape = mean_absolute_percentage_error(filtered_actual, filtered_igou) * 100
        bs_mape = mean_absolute_percentage_error(filtered_actual, filtered_bs) * 100
        bns_mape = mean_absolute_percentage_error(filtered_actual, filtered_bns) * 100
        
        # Direction accuracy (percentage of correct directional predictions)
        def direction_accuracy(actual, predicted):
            if len(actual) <= 1:
                return 50.0  # Default value if there's not enough data
            actual_dir = np.diff(actual) > 0
            pred_dir = np.diff(predicted) > 0
            return np.mean(actual_dir == pred_dir) * 100
        
        igou_dir_acc = direction_accuracy(filtered_actual, filtered_igou)
        bs_dir_acc = direction_accuracy(filtered_actual, filtered_bs)
        bns_dir_acc = direction_accuracy(filtered_actual, filtered_bns)
        
        # Prepare results
        results = {
            'actual_prices': actual_prices,
            'igou_prediction': igou_mean_prediction,
            'bs_prediction': bs_prediction,
            'bns_prediction': np.mean(bns_paths, axis=0),
            'metrics': {
                'igou_rmse': igou_rmse,
                'bs_rmse': bs_rmse,
                'bns_rmse': bns_rmse,
                'igou_mae': igou_mae,
                'bs_mae': bs_mae,
                'bns_mae': bns_mae,
                'igou_mape': igou_mape,
                'bs_mape': bs_mape,
                'bns_mape': bns_mape,
                'igou_dir_acc': igou_dir_acc,
                'bs_dir_acc': bs_dir_acc,
                'bns_dir_acc': bns_dir_acc
            }
        }
        
        return results
    else:
        st.error("Not enough valid data points for evaluation after filtering NaN values.")
        return None

# Function to visualize backtesting results
def plot_backtesting_results(results, n_test):
    """
    Plot the backtesting results.
    
    Parameters:
    results (dict): Dictionary with predictions and metrics
    n_test (int): Number of test data points (calculated from test_percentage)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Vérifier la longueur réelle des données
    actual_len = len(results['actual_prices'])
    x_axis = range(actual_len)
    
    # Plot actual prices
    ax.plot(x_axis, results['actual_prices'], 'k-', lw=2, label='Actual Prices')
    
    # Plot IG-OU prediction
    ax.plot(x_axis, results['igou_prediction'][:actual_len], 'r--', lw=2, label='IG-OU Prediction')
    
    # Plot Black-Scholes prediction
    ax.plot(x_axis, results['bs_prediction'][:actual_len], 'b-.', lw=2, label='Black-Scholes Prediction')
    
    # Plot BNS prediction
    ax.plot(x_axis, results['bns_prediction'][:actual_len], 'g:', lw=2, label='BNS Prediction')
    
    ax.set_title('Backtesting Results: Prediction vs Actual Prices')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    return fig

# Add this to the main function
def add_backtesting_section(data):
    """
    Add the backtesting section to the Streamlit app.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with 'Price' price column
    """
    st.header("Model Backtesting")
    
    # Slider for test percentage
    test_percentage = st.slider("Percentage of data to test", 10, 60, 30, step=5) / 100
    
    # Number of simulations for Monte Carlo
    n_simulations = st.number_input("Number of simulations", 10, 1000, 100)
    
    # Check if we have enough data
    if len(data) <= int(len(data) * test_percentage):
        st.error(f"Not enough data for backtesting. Need more than {int(len(data) * test_percentage)} data points.")
        return
    
    # Run backtesting
    with st.spinner("Running backtesting..."):
        results = perform_backtesting(data, test_percentage, n_simulations)
    
    # Check if results is not None
    if results is None:
        return
    
    # Plot results
    fig = plot_backtesting_results(results, n_test=int(len(data) * test_percentage))
    st.pyplot(fig)
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE (Root Mean Squared Error)', 
                   'MAE (Mean Absolute Error)', 
                   'MAPE (Mean Absolute Percentage Error)',
                   'Direction Accuracy (%)'],
        'IG-OU Model': [results['metrics']['igou_rmse'], 
                       results['metrics']['igou_mae'], 
                       results['metrics']['igou_mape'],
                       results['metrics']['igou_dir_acc']],
        'Black-Scholes Model': [results['metrics']['bs_rmse'], 
                               results['metrics']['bs_mae'], 
                               results['metrics']['bs_mape'],
                               results['metrics']['bs_dir_acc']],
        'BNS Model': [results['metrics']['bns_rmse'], 
                      results['metrics']['bns_mae'], 
                      results['metrics']['bns_mape'],
                      results['metrics']['bns_dir_acc']]
    })
    
    # Format columns with float values
    for col in ['IG-OU Model', 'Black-Scholes Model', 'BNS Model']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}")
    
    st.table(metrics_df)
    
    # Determine the best model based on RMSE
    best_rmse = min(results['metrics']['igou_rmse'], results['metrics']['bs_rmse'], results['metrics']['bns_rmse'])
    best_model = "IG-OU" if results['metrics']['igou_rmse'] == best_rmse else "Black-Scholes" if results['metrics']['bs_rmse'] == best_rmse else "BNS"
    
    st.success(f"Based on RMSE, the {best_model} model performed better for this time period.")
    
    # Add interpretation
    with st.expander("Interpretation of Metrics"):
        st.markdown("""
        ### Metrics Explanation:
        
        - **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared differences between predicted and actual values. 
          Lower values indicate better performance. RMSE penalizes large errors more than small ones.
        
        - **MAE (Mean Absolute Error)**: Measures the average absolute differences between predicted and actual values. 
          Easier to interpret as it's in the same units as the data. Lower values indicate better performance.
        
        - **MAPE (Mean Absolute Percentage Error)**: Expresses error as a percentage of the actual values, making it scale-independent.
          Lower values indicate better performance. A MAPE of 5% means predictions are on average 5% away from actual values.
        
        - **Direction Accuracy**: Percentage of times the model correctly predicted the direction of price movement (up or down).
          Higher values indicate better performance. A value of 50% is no better than random guessing.
        
        ### What to Look For:
        
        - The model with lower RMSE, MAE, and MAPE values generally performs better at capturing the price levels.
        - Direction accuracy is especially important if you're more concerned with predicting price movements than exact values.
        - Consider the tradeoffs between metrics based on your investment strategy.
        """)
    
    # Show raw data
    with st.expander("Show Raw Prediction Data"):
        prediction_df = pd.DataFrame({
            'Day': range(1, int(len(data) * test_percentage) + 1),
            'Actual Price': results['actual_prices'],
            'IG-OU Prediction': results['igou_prediction'],
            'Black-Scholes Prediction': results['bs_prediction'],
            'BNS Prediction': results['bns_prediction'],
            'IG-OU Error (%)': ((results['igou_prediction'] - results['actual_prices']) / results['actual_prices'] * 100),
            'BS Error (%)': ((results['bs_prediction'] - results['actual_prices']) / results['actual_prices'] * 100),
            'BNS Error (%)': ((results['bns_prediction'] - results['actual_prices']) / results['actual_prices'] * 100)
        })
        st.dataframe(prediction_df)

class VolatilityPlotter:
    @staticmethod
    def plot_acf(series, title):
        """Autocorrelation Function plot"""
        plt.figure(figsize=(12,6))
        pd.plotting.autocorrelation_plot(series)
        plt.title(title)
        return plt.gcf()

    @staticmethod
    def plot_pdf_comparison(data, inv_gaussian_params, title):
        """Compare empirical PDF with Inverse Gaussian"""
        plt.figure(figsize=(12,6))
        sns.histplot(data, kde=True, stat='density')
        x = np.linspace(min(data), max(data), 100)
        plt.plot(x, invgauss.pdf(x, *inv_gaussian_params), 'r-')
        plt.title(title)
        return plt.gcf()

    @staticmethod
    def plot_residual_analysis(residuals, title):
        """ACF and histogram of squared residuals"""
        fig, ax = plt.subplots(2,1, figsize=(12,12))
        
        # ACF des résidus carrés
        sm.graphics.tsa.plot_acf(residuals**2, ax=ax[0])
        ax[0].set_title(f'ACF des résidus carrés - {title}')
        
        # Histogramme
        sns.histplot(residuals, kde=True, ax=ax[1])
        ax[1].set_title(f'Distribution des résidus - {title}')
        
        return fig

    @staticmethod
    def plot_confidence_intervals(actual, predictions, title):
        """Visualize model confidence intervals"""
        plt.figure(figsize=(12,6))
        plt.plot(actual, 'b-', label='Actual Prices')
        plt.plot(np.mean(predictions, axis=0), 'r--', label='Mean Prediction')
        plt.fill_between(range(len(actual)),
                        np.percentile(predictions, 5, axis=0),
                        np.percentile(predictions, 95, axis=0),
                        color='red', alpha=0.2, label='90% Confidence Band')
        plt.title(title)
        plt.legend()
        return plt.gcf()

def calculate_model_metrics(actual, predicted):
    """Calculate comprehensive model diagnostics"""
    residuals = actual - predicted
    return {
        'shapiro_stat': shapiro(residuals)[0],
        'shapiro_p': shapiro(residuals)[1],
        'arch_lm_stat': arch_lm_test(residuals)[0],
        'arch_lm_p': arch_lm_test(residuals)[1],
        'jb_stat': jarque_bera(residuals)[0],
        'jb_p': jarque_bera(residuals)[1]
    }

def arch_lm_test(residuals, lags=5):
    """ARCH-LM test for volatility clustering"""
    return sm.stats.diagnostic.het_arch(residuals, maxlag=lags)

def add_diagnostic_section(data):
    """New section for model diagnostics"""
    st.header("Analyse Diagnostique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox("Modèle", ["IG-OU", "Black-Scholes", "BNS"])
        
    with col2:
        plot_type = st.selectbox("Type de visualisation",
                               ["ACF des prix", 
                                "PDF des rendements",
                                "Intervalles de confiance",
                                "Analyse des résidus"])
    
    if selected_model == "IG-OU":
        model_data = data['Price']
    elif selected_model == "Black-Scholes":
        model_data = simulate_bs(data['Price'].iloc[-1], 0.05, 0.2)
    else:
        model_data = simulate_bns(data['Price'].iloc[-1], 0.1, 2.2395e-7, 1.0, 0.1)
    
    if plot_type == "ACF des prix":
        fig = VolatilityPlotter.plot_acf(model_data, f"ACF - {selected_model}")
    elif plot_type == "PDF des rendements":
        returns = pd.Series(model_data).pct_change().dropna()
        fig = VolatilityPlotter.plot_pdf_comparison(returns, (0, returns.std()), "PDF des Rendements")
    elif plot_type == "Intervalles de confiance":
        fig = VolatilityPlotter.plot_confidence_intervals(data['Price'].values[-30:], 
                                                         np.array([model_data]*100), 
                                                         "Intervalles de Confiance")
    else:
        residuals = data['Price'].values[-len(model_data):] - model_data
        fig = VolatilityPlotter.plot_residual_analysis(residuals, selected_model)
    
    st.pyplot(fig)
    
    if st.button("Exécuter les tests statistiques"):
        metrics = calculate_model_metrics(data['Price'].values[-len(model_data):], model_data)
        st.subheader("Résultats des tests")
        st.write(f"Shapiro-Wilk (Normalité): W={metrics['shapiro_stat']:.3f}, p={metrics['shapiro_p']:.4f}")
        st.write(f"ARCH-LM (Clustering de volatilité): LM={metrics['arch_lm_stat']:.2f}, p={metrics['arch_lm_p']:.4f}")
        st.write(f"Jarque-Bera (Normalité): JB={metrics['jb_stat']:.2f}, p={metrics['jb_p']:.4f}")

def main():
    st.title("Prédiction de Prix et Volatilité")
    
    # Sélection des données
    data_source = st.selectbox("Source des données", ["Yahoo Finance", "Fichier Excel/CSV", "Données d'exemple"])
    
    # Initialisation des variables
    data = None
    returns = None
    historical_prices = pd.Series()
    
    if data_source == "Yahoo Finance":
        with st.form("yahoo_form"):
            ticker = st.text_input("Symbole Yahoo Finance (ex: AAPL)", "AAPL")
            period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            submit_button = st.form_submit_button("Télécharger")
            
            if submit_button:
                # Afficher un message de chargement
                with st.spinner('Téléchargement des données...'):
                    try:
                        # Utiliser des proxies et multiple tentatives
                        for attempt in range(3):
                            try:
                                # Utiliser des proxies et multiple tentatives
                                data = yf.download(ticker, period=period, progress=False)
                                if not data.empty and 'Price' in data.columns:
                                    break
                            except Exception as e:
                                st.warning(f"Tentative {attempt+1}: {str(e)}")
                                if attempt < 2:
                                    time.sleep(1)  # Attendre avant de réessayer
                                continue
                        
                        if data is None or data.empty or 'Price' not in data.columns:
                            st.error(f"Impossible de récupérer les données pour {ticker}")
                        else:
                            st.success(f"Données téléchargées pour {ticker}")
                            # Vérifier que Price contient des données numériques
                            if data['Price'].dtype.kind in 'iuf':  # i: int, u: uint, f: float
                                returns = data['Price'].pct_change(fill_method=None).dropna()
                                st.line_chart(data['Price'])
                            else:
                                st.error("Les données ne contiennent pas de valeurs numériques")
                    except Exception as e:
                        st.error(f"Erreur lors du téléchargement: {str(e)}")
    elif data_source == "Fichier Excel/CSV":
        uploaded_file = st.file_uploader("Télécharger fichier", type=["xlsx", "csv"])
        if uploaded_file is not None:
            try:
                # Déterminer le format et lire le fichier
                if uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:  # CSV
                    # Essayer différentes options de parsing pour CSV
                    try:
                        # Essayer d'abord le format standard
                        data = pd.read_csv(uploaded_file)
                    except:
                        # Réinitialiser le curseur du fichier
                        uploaded_file.seek(0)
                        # Essayer avec différents séparateurs et la gestion des milliers
                        data = pd.read_csv(uploaded_file, sep=None, engine='python', thousands=',')
                
                # Afficher les premières lignes pour vérification
                st.write("Aperçu des données:")
                st.write(data.head())
                
                # Vérifier que la colonne 'Price' existe
                if 'Price' not in data.columns:
                    st.warning("La colonne 'Price' n'a pas été trouvée. Sélectionnez la colonne contenant les prix:")
                    price_column = st.selectbox("Colonne des prix", data.columns)
                    if price_column:
                        # Créer une colonne 'Price' à partir de la colonne sélectionnée
                        data['Price'] = pd.to_numeric(data[price_column], errors='coerce')
                
                # Convertir en nombres si nécessaire
                if 'Price' in data.columns:
                    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
                    # Vérifier et alerter l'utilisateur sur les valeurs manquantes
                    missing = data['Price'].isna().sum()
                    if missing > 0:
                        st.warning(f"{missing} valeurs manquantes trouvées et ignorées.")
                    
                    # Calculer les rendements en ignorant les valeurs manquantes
                    returns = data['Price'].dropna().pct_change(fill_method=None).dropna()
                    
                    if len(returns) > 0:
                        st.line_chart(data['Price'].dropna())
                    else:
                        st.error("Impossible de calculer les rendements. Données insuffisantes.")
                else:
                    st.error("Impossible de déterminer les prix de clôture")
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
    else:  # Données d'exemple
        with st.spinner("Chargement des données d'exemple..."):
            try:
                # Essayer de charger le fichier BTC-USD.csv
                data = pd.read_csv('BTC-USD.csv', parse_dates=['Date'], index_col='Date')
                if 'Price' in data.columns:
                    returns = data['Price'].pct_change(fill_method=None).dropna()
                    st.success("Données d'exemple chargées avec succès")
                    st.line_chart(data['Price'])
                else:
                    st.error("Format de données inattendu (pas de colonne 'Price')")
            except FileNotFoundError:
                st.error("Le fichier BTC-USD.csv n'a pas été trouvé")
                # Créer des données d'exemple
                dates = pd.date_range(start=datetime.now()-timedelta(days=365), periods=365)
                Price_prices = np.cumsum(np.random.normal(0, 1, 365)) + 100
                data = pd.DataFrame({'Price': Price_prices}, index=dates)
                returns = data['Price'].pct_change(fill_method=None).dropna()
                st.success("Données d'exemple générées avec succès")
                st.line_chart(data['Price'])
            except Exception as e:
                st.error(f"Erreur lors du chargement des données d'exemple: {str(e)}")
    
    # Vérifier que nous avons des données à traiter
    if data is not None and returns is not None and not returns.empty:
        # Afficher les statistiques des données
        st.subheader("Statistiques des rendements")
        stats_df = pd.DataFrame({
            'Moyenne': [returns.mean()],
            'Écart-type': [returns.std()],
            'Min': [returns.min()],
            'Max': [returns.max()]
        })
        st.write(stats_df)
        
        # Add backtesting section
        add_backtesting_section(data)
        
        # Estimation des paramètres
        mu, sigma_sq, lambda_ = estimate_parameters(returns)
        
        # Paramètres de simulation
        st.sidebar.header("Paramètres de simulation")
        n_simulations = st.sidebar.number_input("Nombre de simulations", 10, 1000, 100)
        a = st.sidebar.number_input("Paramètre a (IG)", 1e-10, 10.0, 2.2395e-7, format="%.7e")
        b = st.sidebar.number_input("Paramètre b (IG)", 0.1, 10.0, 1.0)
        
        # Modèles à utiliser
        st.sidebar.header("Modèles de prédiction")
        use_igou = st.sidebar.checkbox("Modèle IG-OU", value=True)
        use_bs = st.sidebar.checkbox("Modèle Black-Scholes", value=True)
        use_bns = st.sidebar.checkbox("Modèle BNS", value=True)
        
        # Simulation de la volatilité et des prix
        vol_paths = np.zeros((n_simulations, 30))
        price_paths = np.zeros((n_simulations, 30))
        bns_price_paths = np.zeros((n_simulations, 30))
        
        # Vérifier que l'accès à la dernière valeur est sûr
        if data['Price'].dropna().shape[0] > 0:
            last_price = data['Price'].dropna().iloc[-1]
            
            # Éviter les valeurs extrêmes pour la volatilité initiale
            init_vol = max(min(returns.std(), 0.05), 0.001)
            
            # Essayer de charger les données historiques
            try:
                historical_data = pd.read_csv('BTC-USD.csv', parse_dates=['Date'], index_col='Date')
                historical_prices = historical_data['Price'].iloc[-60:]  # 60 dernières valeurs
            except Exception:
                historical_prices = pd.Series()
            
            if use_igou:
                for i in range(n_simulations):
                    try:
                        vol = simulate_ig_ou(X0=init_vol,
                                            lambda_=max(lambda_, 0.001),
                                            a=a, 
                                            b=b, 
                                            T=30)
                        
                        # Simulation des prix
                        prices = [last_price]
                        for t in range(29):
                            drift = mu  # Simplification pour éviter vol[t]**2 qui peut causer des problèmes
                            shock = vol[t] * np.random.normal()
                            prices.append(prices[-1] * np.exp(drift + shock))
                        
                        vol_paths[i] = vol[:30]
                        price_paths[i] = prices
                    except Exception as e:
                        st.error(f"Erreur simulation IG-OU {i+1}: {str(e)}")
                        continue

            # Simulation Black-Scholes
            if use_bs:
                bs_prices = simulate_bs(last_price, mu, returns.std(), days=30)
            else:
                bs_prices = None
            
            # Simulation BNS
            if use_bns:
                bns_prices = simulate_bns(last_price, lambda_, a=a, b=b, theta=0.1, T=30)
                for i in range(n_simulations):
                    bns_price_paths[i] = bns_prices
            else:
                bns_prices = None
            
            # Visualisation
            st.subheader("Résultats de simulation")
            
            # Configuration des graphiques selon les modèles sélectionnés
            if use_igou and use_bs and use_bns:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                
                # Graphique de prix combiné
                if not historical_prices.empty:
                    # Si nous avons des données historiques, les afficher
                    dates = pd.date_range(end=datetime.now(), periods=len(historical_prices))
                    ax1.plot(dates, historical_prices, 'b-', lw=2, label='Données historiques')
                
                # Simulation IG-OU
                for path in price_paths:
                    ax1.plot(range(30), path, lw=1, alpha=0.1, color='red')
                mean_price = np.mean(price_paths, axis=0)
                ax1.plot(range(30), mean_price, 'r--', lw=2, label='Prédiction IG-OU (Moyenne)')
                
                # Simulation Black-Scholes
                ax1.plot(range(30), bs_prices, 'orange', lw=2, label='Prédiction Black-Scholes')
                
                # Simulation BNS
                ax1.plot(range(30), bns_prices, 'green', lw=2, label='Prédiction BNS')
                
                ax1.set_title('Comparaison des modèles sur 30 jours')
                ax1.legend()
                ax1.grid(True)
                
                # Graphique de volatilité
                for path in vol_paths:
                    ax2.plot(path, lw=1, alpha=0.1, color='green')
                mean_vol = np.mean(vol_paths, axis=0)
                ax2.plot(mean_vol, lw=2, color='green', label='Volatilité moyenne')
                ax2.set_title('Projection de la volatilité sur 30 jours (IG-OU)')
                ax2.legend()
                ax2.grid(True)
                
            elif use_igou and use_bs:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                
                # Graphique de prix combiné
                if not historical_prices.empty:
                    # Si nous avons des données historiques, les afficher
                    dates = pd.date_range(end=datetime.now(), periods=len(historical_prices))
                    ax1.plot(dates, historical_prices, 'b-', lw=2, label='Données historiques')
                
                # Simulation IG-OU
                for path in price_paths:
                    ax1.plot(range(30), path, lw=1, alpha=0.1, color='red')
                mean_price = np.mean(price_paths, axis=0)
                ax1.plot(range(30), mean_price, 'r--', lw=2, label='Prédiction IG-OU (Moyenne)')
                
                # Simulation Black-Scholes
                ax1.plot(range(30), bs_prices, 'orange', lw=2, label='Prédiction Black-Scholes')
                
                ax1.set_title('Comparaison des modèles sur 30 jours')
                ax1.legend()
                ax1.grid(True)
                
                # Graphique de volatilité
                for path in vol_paths:
                    ax2.plot(path, lw=1, alpha=0.1, color='green')
                mean_vol = np.mean(vol_paths, axis=0)
                ax2.plot(mean_vol, lw=2, color='green', label='Volatilité moyenne')
                ax2.set_title('Projection de la volatilité sur 30 jours (IG-OU)')
                ax2.legend()
                ax2.grid(True)
                
            elif use_igou and use_bns:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                
                # Graphique de prix combiné
                if not historical_prices.empty:
                    # Si nous avons des données historiques, les afficher
                    dates = pd.date_range(end=datetime.now(), periods=len(historical_prices))
                    ax1.plot(dates, historical_prices, 'b-', lw=2, label='Données historiques')
                
                # Simulation IG-OU
                for path in price_paths:
                    ax1.plot(range(30), path, lw=1, alpha=0.1, color='red')
                mean_price = np.mean(price_paths, axis=0)
                ax1.plot(range(30), mean_price, 'r--', lw=2, label='Prédiction IG-OU (Moyenne)')
                
                # Simulation BNS
                ax1.plot(range(30), bns_prices, 'green', lw=2, label='Prédiction BNS')
                
                ax1.set_title('Comparaison des modèles sur 30 jours')
                ax1.legend()
                ax1.grid(True)
                
                # Graphique de volatilité
                for path in vol_paths:
                    ax2.plot(path, lw=1, alpha=0.1, color='green')
                mean_vol = np.mean(vol_paths, axis=0)
                ax2.plot(mean_vol, lw=2, color='green', label='Volatilité moyenne')
                ax2.set_title('Projection de la volatilité sur 30 jours (IG-OU)')
                ax2.legend()
                ax2.grid(True)
                
            elif use_bs and use_bns:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Graphique de prix combiné
                if not historical_prices.empty:
                    # Si nous avons des données historiques, les afficher
                    dates = pd.date_range(end=datetime.now(), periods=len(historical_prices))
                    ax.plot(dates, historical_prices, 'b-', lw=2, label='Données historiques')
                
                # Simulation Black-Scholes
                ax.plot(range(30), bs_prices, 'orange', lw=2, label='Prédiction Black-Scholes')
                
                # Simulation BNS
                ax.plot(range(30), bns_prices, 'green', lw=2, label='Prédiction BNS')
                
                ax.set_title('Comparaison des modèles sur 30 jours')
                ax.legend()
                ax.grid(True)
                
            elif use_igou:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                
                # Prix IG-OU
                for path in price_paths:
                    ax1.plot(path, lw=1, alpha=0.1, color='blue')
                mean_price = np.mean(price_paths, axis=0)
                ax1.plot(mean_price, lw=2, color='blue', label='Moyenne')
                ax1.set_title('Projection des prix sur 30 jours (IG-OU)')
                ax1.legend()
                ax1.grid(True)
                
                # Volatilité IG-OU
                for path in vol_paths:
                    ax2.plot(path, lw=1, alpha=0.1, color='red')
                mean_vol = np.mean(vol_paths, axis=0)
                ax2.plot(mean_vol, lw=2, color='red', label='Moyenne')
                ax2.set_title('Projection de la volatilité sur 30 jours (IG-OU)')
                ax2.legend()
                ax2.grid(True)
                
            elif use_bs:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(bs_prices, 'orange', lw=2, label='Prédiction Black-Scholes')
                ax.set_title('Projection des prix sur 30 jours (Black-Scholes)')
                ax.legend()
                ax.grid(True)
                
            elif use_bns:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(bns_prices, 'green', lw=2, label='Prédiction BNS')
                ax.set_title('Projection des prix sur 30 jours (BNS)')
                ax.legend()
                ax.grid(True)
            else:
                st.warning("Aucun modèle sélectionné pour la simulation")
                fig = None
            
            if fig:
                st.pyplot(fig)
            
            # Afficher tableau avec valeurs de prédiction
            st.subheader("Prévisions numériques")
            
            days = list(range(1, 31))
            forecast_data = {'Jour': days}
            
            if use_igou:
                mean_price_igou = np.mean(price_paths, axis=0)
                mean_vol_igou = np.mean(vol_paths, axis=0)
                forecast_data['Prix moyen (IG-OU)'] = mean_price_igou
                forecast_data['Volatilité moyenne (IG-OU)'] = mean_vol_igou
                
            if use_bs:
                forecast_data['Prix (Black-Scholes)'] = bs_prices
            
            if use_bns:
                forecast_data['Prix (BNS)'] = bns_prices
            
            forecast_df = pd.DataFrame(forecast_data)
            st.dataframe(forecast_df)
            
            # Statistiques et interprétation
            st.subheader("Statistiques estimées")
            st.write(f"Moyenne estimée (μ): {mu:.6f}")
            st.write(f"Variance estimée (σ²): {sigma_sq:.6f}")
            st.write(f"Lambda estimé (λ): {lambda_:.4f}")
            
            # Explication des modèles
            with st.expander("À propos des modèles utilisés"):
                st.markdown("""
                ### Modèle IG-OU
                Le modèle IG-OU (Inverse Gaussian - Ornstein-Uhlenbeck) est une extension du modèle de volatilité stochastique. 
                Il utilise la distribution Inverse Gaussienne pour modéliser la volatilité, ce qui permet de capturer les sauts et 
                les distributions asymétriques des rendements financiers.
                
                ### Modèle Black-Scholes
                Le modèle Black-Scholes est un modèle classique en finance qui suppose que les prix des actifs suivent un mouvement 
                brownien géométrique avec volatilité constante. Il repose sur l'hypothèse que les marchés sont efficaces et 
                que les rendements sont normalement distribués.
                
                ### Modèle BNS
                Le modèle BNS (Barndorff-Nielsen & Shephard) est un modèle de volatilité stochastique qui utilise la distribution 
                Inverse Gaussienne pour modéliser la volatilité. Il est similaire au modèle IG-OU mais utilise une approche différente 
                pour modéliser la volatilité.
                """)
        else:
            st.error("Données insuffisantes pour la simulation")
    
    # Add diagnostic section
    add_diagnostic_section(data)

if __name__ == "__main__":
    # Configuration de la page
    st.set_page_config(
        page_title="Prédiction Prix & Volatilité",
        page_icon="📈",
        layout="wide"
    )
    main()