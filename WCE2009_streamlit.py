import time
from datetime import datetime, timedelta
from io import StringIO
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)


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
    
    return X[:30]  # Assurons-nous de renvoyer exactement 30 valeurs

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
            rho1 = 0.1  # En cas d'erreur
    else:
        rho1 = 0.1
    
    # Éviter un lambda négatif ou zéro
    if rho1 >= 0.999:
        rho1 = 0.999
    elif rho1 <= 0:
        rho1 = 0.001
        
    lambda_ = -np.log(max(rho1, 1e-6))
    
    return mu, sigma_sq, lambda_

# Modified function to perform backtesting of prediction models
def perform_backtesting(data, n_test=30, n_simulations=100, test_data=None):
    """
    Performs backtesting of prediction models by using historical data to predict future data points.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with 'Close' price column (used as training data)
    n_test (int): Number of data points to use for testing when test_data is None
    n_simulations (int): Number of Monte Carlo simulations to run
    test_data (pandas.DataFrame, optional): Specific test data to use. If None, the last n_test points of data are used
    
    Returns:
    dict: Dictionary with predictions and metrics for both models
    """
    # Ensure data is sorted by date
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    
    # Remove NaN values from the data before splitting
    data = data.dropna(subset=['Close'])
    
    # If test_data is provided, use it directly. Otherwise extract from data
    if test_data is None:
        # Check if we have enough data after dropping NaNs
        if len(data) <= n_test:
            st.error(f"Not enough valid data for backtesting. Need more than {n_test} non-NaN data points.")
            return None
        
        # Split data into training and testing sets
        train_data = data[:-n_test].copy()
        test_data = data[-n_test:].copy()
    else:
        train_data = data.copy()
        # Ensure test_data also has no NaN values
        test_data = test_data.dropna(subset=['Close'])
        n_test = len(test_data)
    
    # Calculate returns for the training set
    returns = train_data['Close'].pct_change(fill_method=None).dropna()
    
    # Estimate parameters from training data
    mu, sigma_sq, lambda_ = estimate_parameters(returns)
    
    # Get the last price from training data for initialization
    last_price = train_data['Close'].iloc[-1]
    init_vol = max(min(returns.std(), 0.05), 0.001)
    
    # Initialize arrays for predictions
    igou_paths = np.zeros((n_simulations, n_test))
    
    # Run IG-OU model simulations
    for i in range(n_simulations):
        try:
            # Simulate volatility
            vol = simulate_ig_ou(X0=init_vol, lambda_=max(lambda_, 0.001), a=2.2395e-7, b=1.0, T=n_test)
            
            # Simulate prices with the simulated volatility
            prices = [last_price]
            for t in range(n_test-1):
                drift = mu
                shock = vol[t] * np.random.normal()
                prices.append(prices[-1] * np.exp(drift + shock))
            
            igou_paths[i] = prices
        except Exception as e:
            st.error(f"Error in IG-OU simulation {i+1}: {str(e)}")
            # Fill with the last successful simulation or with zeros
            igou_paths[i] = igou_paths[max(0, i-1)] if i > 0 else np.zeros(n_test)
    
    # Calculate mean prediction for IG-OU model
    igou_mean_prediction = np.mean(igou_paths, axis=0)
    
    # Run Black-Scholes simulation
    bs_prediction = simulate_bs(last_price, mu, returns.std(), days=n_test)
    
    # Get actual test values
    actual_prices = test_data['Close'].values
    
    # Ensure there are no NaN values in any of the arrays
    valid_indices = ~np.isnan(actual_prices) & ~np.isnan(igou_mean_prediction) & ~np.isnan(bs_prediction)
    
    if np.any(valid_indices):
        # Filter out NaN values
        filtered_actual = actual_prices[valid_indices]
        filtered_igou = igou_mean_prediction[valid_indices]
        filtered_bs = bs_prediction[valid_indices]
        
        # Calculate metrics
        igou_rmse = sqrt(mean_squared_error(filtered_actual, filtered_igou))
        bs_rmse = sqrt(mean_squared_error(filtered_actual, filtered_bs))
        
        igou_mae = mean_absolute_error(filtered_actual, filtered_igou)
        bs_mae = mean_absolute_error(filtered_actual, filtered_bs)
        
        # Mean Absolute Percentage Error (MAPE)
        igou_mape = mean_absolute_percentage_error(filtered_actual, filtered_igou) * 100
        bs_mape = mean_absolute_percentage_error(filtered_actual, filtered_bs) * 100
        
        # Direction accuracy (percentage of correct directional predictions)
        def direction_accuracy(actual, predicted):
            if len(actual) <= 1:
                return 50.0  # Default value if there's not enough data
            actual_dir = np.diff(actual) > 0
            pred_dir = np.diff(predicted) > 0
            return np.mean(actual_dir == pred_dir) * 100
        
        igou_dir_acc = direction_accuracy(filtered_actual, filtered_igou)
        bs_dir_acc = direction_accuracy(filtered_actual, filtered_bs)
        
        # Prepare results
        results = {
            'actual_prices': actual_prices,
            'igou_prediction': igou_mean_prediction,
            'bs_prediction': bs_prediction,
            'metrics': {
                'igou_rmse': igou_rmse,
                'bs_rmse': bs_rmse,
                'igou_mae': igou_mae,
                'bs_mae': bs_mae,
                'igou_mape': igou_mape,
                'bs_mape': bs_mape,
                'igou_dir_acc': igou_dir_acc,
                'bs_dir_acc': bs_dir_acc
            }
        }
        
        return results
    else:
        st.error("Not enough valid data points for evaluation after filtering NaN values.")
        return None
    
# Nouvelle visualisation pour la validation croisée
def plot_cross_validation(results):
    """
    Visualise les résultats de la validation croisée temporelle.
    
    Parameters:
    results (dict): Résultats de la validation croisée temporelle
    
    Returns:
    matplotlib.figure.Figure: Figure matplotlib avec les tracés
    """
    n_splits = len(results['predictions'])
    
    # Gérer le cas avec un seul split
    if n_splits == 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax]
    else:
        # Définition de la taille du graphique en fonction du nombre de splits
        fig_height = 4 * min(n_splits, 5)  # Limiter la hauteur pour beaucoup de splits
        fig, axes = plt.subplots(min(n_splits, 5), 1, figsize=(12, fig_height))
        
        # Si n_splits > 5, on affiche seulement les 5 premiers
        if n_splits > 5:
            st.info(f"Affichage des 5 premiers splits sur {n_splits} pour plus de clarté.")
    
    # Assurez-vous que axes est toujours une liste/array même avec un seul subplot
    if n_splits == 1:
        axes = [axes]
    
    # Boucle sur les premières fenêtres (max 5 pour la clarté)
    for i, split_data in enumerate(results['predictions'][:min(n_splits, 5)]):
        ax = axes[i]
        
        # Tracer les données
        ax.plot(range(len(split_data['actual'])), split_data['actual'], 'k-', label='Prix réels')
        ax.plot(range(len(split_data['igou'])), split_data['igou'], 'r--', label='IG-OU')
        ax.plot(range(len(split_data['bs'])), split_data['bs'], 'b-.', label='BS')
        
        # Calculer l'erreur absolue moyenne pour les deux modèles sur ce split
        igou_rmse = split_data['metrics']['igou_rmse']
        bs_rmse = split_data['metrics']['bs_rmse']
        
        # Déterminer le meilleur modèle pour ce split
        best_model = "IG-OU" if igou_rmse < bs_rmse else "Black-Scholes"
        
        # Titre et légendes
        ax.set_title(f"Split {i+1} | Meilleur: {best_model} | RMSE IG-OU: {igou_rmse:.2f}, BS: {bs_rmse:.2f}")
        ax.grid(True)
        ax.legend(loc='upper left')
        
        # Ajouter des étiquettes aux axes seulement pour le dernier graphique
        if i == min(n_splits, 5) - 1:
            ax.set_xlabel('Jours')
        ax.set_ylabel('Prix')
    
    plt.tight_layout()
    return fig
    
# Function to visualize backtesting results
def plot_backtesting_results(results, n_test=30):
    """
    Plot the backtesting results.
    
    Parameters:
    results (dict): Dictionary with predictions and metrics
    n_test (int): Number of test data points
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot actual prices
    ax.plot(range(n_test), results['actual_prices'], 'k-', lw=2, label='Actual Prices')
    
    # Plot IG-OU prediction
    ax.plot(range(n_test), results['igou_prediction'], 'r--', lw=2, label='IG-OU Prediction')
    
    # Plot Black-Scholes prediction
    ax.plot(range(n_test), results['bs_prediction'], 'b-.', lw=2, label='Black-Scholes Prediction')
    
    ax.set_title('Backtesting Results: Prediction vs Actual Prices')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    return fig

# Nouvelle fonction de validation croisée temporelle
def time_series_cross_validate(data, n_splits=5, test_size=30, n_simulations=100):
    """
    Effectue une validation croisée temporelle avec fenêtre glissante
    
    Paramètres:
    data (pd.DataFrame): Données historiques
    n_splits (int): Nombre de splits de validation
    test_size (int): Taille de la période de test (jours)
    n_simulations (int): Nombre de simulations Monte Carlo
    
    Retourne:
    dict: Métriques agrégées et prédictions
    """
    total_metrics = {
        'igou': {'rmse': [], 'mae': [], 'mape': [], 'dir_acc': []},
        'bs': {'rmse': [], 'mae': [], 'mape': [], 'dir_acc': []}
    }
    all_predictions = []
    
    # Calcul des positions des fenêtres
    data_length = len(data)
    min_train_size = data_length - test_size * n_splits
    if min_train_size < test_size * 2:
        st.error("Pas assez de données pour la validation croisée")
        return None
    
    # Création des fenêtres glissantes
    for i in range(n_splits):
        start_test = data_length - test_size * (i + 1)
        end_test = start_test + test_size
        
        train_data = data.iloc[:start_test]
        test_data = data.iloc[start_test:end_test]
        
        # Entraînement et prédiction
        results = perform_backtesting(train_data, test_size, n_simulations, test_data)
        
        if results:
            for metric in ['rmse', 'mae', 'mape', 'dir_acc']:
                total_metrics['igou'][metric].append(results['metrics'][f'igou_{metric}'])
                total_metrics['bs'][metric].append(results['metrics'][f'bs_{metric}'])
            
            all_predictions.append({
                'split': i+1,
                'actual': results['actual_prices'],
                'igou': results['igou_prediction'],
                'bs': results['bs_prediction']
            })
    
    # Calcul des moyennes
    avg_metrics = {
        'igou': {k: np.nanmean(v) for k, v in total_metrics['igou'].items()},
        'bs': {k: np.nanmean(v) for k, v in total_metrics['bs'].items()}
    }
    
    return {'metrics': avg_metrics, 'predictions': all_predictions}

# Add this to the main function
def add_backtesting_section(data):
    """
    Add the backtesting section to the Streamlit app with time series cross-validation.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with 'Close' price column
    """
    st.header("Validation croisée temporelle")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_splits = st.number_input("Nombre de splits", 2, 10, 3)
    with col2:
        test_size = st.number_input("Taille de test (jours)", 10, 90, 30)
    with col3:
        n_simulations = st.number_input("Simulations par split", 50, 1000, 200)
    
    if len(data) < test_size * (n_splits + 1):
        st.error(f"Besoin d'au moins {test_size * (n_splits + 1)} jours de données")
        return
    
    if st.button("Lancer la validation croisée"):
        with st.spinner("Validation croisée en cours..."):
            cv_results = time_series_cross_validate(
                data, 
                n_splits=n_splits,
                test_size=test_size,
                n_simulations=n_simulations
            )
        
        if cv_results:
            st.subheader("Performance moyenne")
            metrics_df = pd.DataFrame({
                'Modèle': ['IG-OU', 'Black-Scholes'],
                'RMSE': [
                    cv_results['metrics']['igou']['rmse'],
                    cv_results['metrics']['bs']['rmse']
                ],
                'MAE': [
                    cv_results['metrics']['igou']['mae'],
                    cv_results['metrics']['bs']['mae']
                ],
                'MAPE (%)': [
                    cv_results['metrics']['igou']['mape'],
                    cv_results['metrics']['bs']['mape']
                ],
                'Précision direction (%)': [
                    cv_results['metrics']['igou']['dir_acc'],
                    cv_results['metrics']['bs']['dir_acc']
                ]
            }).set_index('Modèle')
            
            st.dataframe(metrics_df.style.format("{:.2f}"))
            
            # Déterminer le meilleur modèle basé sur le RMSE moyen
            best_rmse = min(cv_results['metrics']['igou']['rmse'], cv_results['metrics']['bs']['rmse'])
            best_model = "IG-OU" if cv_results['metrics']['igou']['rmse'] == best_rmse else "Black-Scholes"
            
            st.success(f"Sur l'ensemble des {n_splits} splits, le modèle {best_model} a obtenu les meilleures performances (RMSE moyen: {best_rmse:.4f}).")
            
            st.subheader("Détails par split")
            fig = plot_cross_validation(cv_results)
            st.pyplot(fig)
            
            # Ajouter l'analyse de stabilité
            st.subheader("Analyse de stabilité")
            rmse_values = [
                (split['metrics']['igou_rmse'], split['metrics']['bs_rmse']) 
                for split in cv_results['predictions']
            ]
            rmse_df = pd.DataFrame(rmse_values, columns=['IG-OU', 'BS'], 
                                 index=range(1, len(rmse_values)+1))
            
            st.line_chart(rmse_df, use_container_width=True)
            st.caption("Évolution des RMSE sur les différents splits")
            
            # Interprétation de la stabilité
            with st.expander("Interprétation de la stabilité des modèles"):
                st.markdown("""
                ### Analyse de la stabilité des modèles
                
                La stabilité d'un modèle prédictif est aussi importante que sa précision moyenne. Un modèle stable 
                conserve une performance similaire à travers différentes périodes de temps, ce qui indique qu'il 
                généralise bien aux nouvelles données.
                
                #### Comment interpréter le graphique de stabilité:
                
                - **Ligne horizontale stable**: Indique que le modèle a une performance constante à travers les différentes périodes.
                - **Fortes variations**: Suggèrent que le modèle est sensible aux changements dans les données et pourrait être moins fiable.
                - **Tendance à la hausse**: Le modèle se dégrade progressivement à mesure qu'on recule dans le temps (potentiellement dû à des changements de régime).
                - **Tendance à la baisse**: Le modèle s'améliore sur les données plus anciennes (rare, mais peut indiquer des conditions plus prévisibles dans le passé).
                
                Le modèle le plus fiable est généralement celui qui combine une bonne performance moyenne avec une faible variance entre les splits.
                """)
            
            # Afficher les prédictions détaillées pour un split sélectionné
            st.subheader("Examiner un split spécifique")
            selected_split = st.selectbox(
                "Sélectionner un split à visualiser en détail",
                options=range(1, len(cv_results['predictions']) + 1)
            )
            
            if selected_split:
                split_data = cv_results['predictions'][selected_split-1]
                
                # Afficher les métriques pour ce split
                st.write(f"### Métriques pour le split {selected_split}")
                split_metrics = pd.DataFrame({
                    'Métrique': ['RMSE', 'MAE', 'MAPE (%)', 'Précision direction (%)'],
                    'IG-OU': [
                        split_data['metrics']['igou_rmse'],
                        split_data['metrics']['igou_mae'],
                        split_data['metrics']['igou_mape'],
                        split_data['metrics']['igou_dir_acc']
                    ],
                    'Black-Scholes': [
                        split_data['metrics']['bs_rmse'],
                        split_data['metrics']['bs_mae'],
                        split_data['metrics']['bs_mape'],
                        split_data['metrics']['bs_dir_acc']
                    ]
                })
                
                # Formater les colonnes numériques
                for col in ['IG-OU', 'Black-Scholes']:
                    split_metrics[col] = split_metrics[col].apply(lambda x: f"{x:.4f}")
                
                st.table(split_metrics)
                
                # Visualiser les prédictions pour ce split
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(range(len(split_data['actual'])), split_data['actual'], 'k-', 
                      label='Prix réels')
                ax.plot(range(len(split_data['igou'])), split_data['igou'], 'r--', 
                      label='Prédiction IG-OU')
                ax.plot(range(len(split_data['bs'])), split_data['bs'], 'b-.', 
                      label='Prédiction Black-Scholes')
                ax.set_title(f'Prédictions pour le split {selected_split}')
                ax.set_xlabel('Jours')
                ax.set_ylabel('Prix')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                
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
                                data = yf.download(ticker, period=period, progress=False)
                                if not data.empty and 'Close' in data.columns:
                                    break
                            except Exception as e:
                                st.warning(f"Tentative {attempt+1}: {str(e)}")
                                if attempt < 2:
                                    time.sleep(1)  # Attendre avant de réessayer
                                continue
                        
                        if data is None or data.empty or 'Close' not in data.columns:
                            st.error(f"Impossible de récupérer les données pour {ticker}")
                        else:
                            st.success(f"Données téléchargées pour {ticker}")
                            # Vérifier que Close contient des données numériques
                            if data['Close'].dtype.kind in 'iuf':  # i: int, u: uint, f: float
                                returns = data['Close'].pct_change(fill_method=None).dropna()
                                st.line_chart(data['Close'])
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
                
                # Vérifier que la colonne 'Close' existe
                if 'Close' not in data.columns:
                    st.warning("La colonne 'Close' n'a pas été trouvée. Sélectionnez la colonne contenant les prix:")
                    price_column = st.selectbox("Colonne des prix", data.columns)
                    if price_column:
                        # Créer une colonne 'Close' à partir de la colonne sélectionnée
                        data['Close'] = pd.to_numeric(data[price_column], errors='coerce')
                
                # Convertir en nombres si nécessaire
                if 'Close' in data.columns:
                    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
                    # Vérifier et alerter l'utilisateur sur les valeurs manquantes
                    missing = data['Close'].isna().sum()
                    if missing > 0:
                        st.warning(f"{missing} valeurs manquantes trouvées et ignorées.")
                    
                    # Calculer les rendements en ignorant les valeurs manquantes
                    returns = data['Close'].dropna().pct_change(fill_method=None).dropna()
                    
                    if len(returns) > 0:
                        st.line_chart(data['Close'].dropna())
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
                if 'Close' in data.columns:
                    returns = data['Close'].pct_change(fill_method=None).dropna()
                    st.success("Données d'exemple chargées avec succès")
                    st.line_chart(data['Close'])
                else:
                    st.error("Format de données inattendu (pas de colonne 'Close')")
            except FileNotFoundError:
                st.error("Le fichier BTC-USD.csv n'a pas été trouvé")
                # Créer des données d'exemple
                dates = pd.date_range(start=datetime.now()-timedelta(days=365), periods=365)
                close_prices = np.cumsum(np.random.normal(0, 1, 365)) + 100
                data = pd.DataFrame({'Close': close_prices}, index=dates)
                returns = data['Close'].pct_change(fill_method=None).dropna()
                st.success("Données d'exemple générées avec succès")
                st.line_chart(data['Close'])
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
        
        # Simulation de la volatilité et des prix
        vol_paths = np.zeros((n_simulations, 30))
        price_paths = np.zeros((n_simulations, 30))
        
        # Vérifier que l'accès à la dernière valeur est sûr
        if data['Close'].dropna().shape[0] > 0:
            last_price = data['Close'].dropna().iloc[-1]
            
            # Éviter les valeurs extrêmes pour la volatilité initiale
            init_vol = max(min(returns.std(), 0.05), 0.001)
            
            # Essayer de charger les données historiques
            try:
                historical_data = pd.read_csv('BTC-USD.csv', parse_dates=['Date'], index_col='Date')
                historical_prices = historical_data['Close'].iloc[-60:]  # 60 dernières valeurs
            except Exception:
                historical_prices = pd.Series()
            
            if use_igou:
                for i in range(n_simulations):
                    try:
                        vol = simulate_ig_ou(X0=init_vol,
                                            lambda_=max(lambda_, 0.001),
                                            a=a, 
                                            b=b)
                        
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
            
            # Visualisation
            st.subheader("Résultats de simulation")
            
            # Configuration des graphiques selon les modèles sélectionnés
            if use_igou and use_bs:
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
                """)
        else:
            st.error("Données insuffisantes pour la simulation")

if __name__ == "__main__":
    # Configuration de la page
    st.set_page_config(
        page_title="Prédiction Prix & Volatilité",
        page_icon="📈",
        layout="wide"
    )
    main()