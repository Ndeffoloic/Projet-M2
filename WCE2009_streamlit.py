import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def generate_ig(a, b, size):
    """Génère des variables aléatoires Inverse Gaussiennes (Algorithme du document)"""
    N = np.random.normal(0, 1, size)
    Y = N**2
    X1 = (a/b) + Y/(2*b**2) - np.sqrt(4*a*b*Y + Y**2)/(2*b**2)
    U = np.random.uniform(0, 1, size)
    mask = U <= a/(a + X1*b)
    X = np.where(mask, X1, a**2/(b**2*X1))
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

def main():
    st.title("Prédiction de Prix et Volatilité")
    
    # Sélection des données
    data_source = st.selectbox("Source des données", ["Yahoo Finance", "Fichier Excel/CSV"])
    
    # Initialisation des variables
    data = None
    returns = None
    
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
    else:
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
        
        # Estimation des paramètres
        mu, sigma_sq, lambda_ = estimate_parameters(returns)
        
        # Paramètres de simulation
        st.sidebar.header("Paramètres de simulation")
        n_simulations = st.sidebar.number_input("Nombre de simulations", 10, 1000, 100)
        a = st.sidebar.number_input("Paramètre a (IG)", 1e-10, 10.0, 2.2395e-7, format="%.7e")
        b = st.sidebar.number_input("Paramètre b (IG)", 0.1, 10.0, 1.0)
        
        # Simulation de la volatilité
        vol_paths = np.zeros((n_simulations, 30))
        price_paths = np.zeros((n_simulations, 30))
        
        # Vérifier que l'accès à la dernière valeur est sûr
        if data['Close'].dropna().shape[0] > 0:
            last_price = data['Close'].dropna().iloc[-1]
            
            # Éviter les valeurs extrêmes pour la volatilité initiale
            init_vol = max(min(returns.std(), 0.05), 0.001)
            
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
                    st.error(f"Erreur simulation {i+1}: {str(e)}")
                    continue
            
            # Visualisation
            st.subheader("Résultats de simulation")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Prix
            for path in price_paths:
                ax1.plot(path, lw=1, alpha=0.1, color='blue')
            # Ajouter la moyenne des chemins
            mean_price = np.mean(price_paths, axis=0)
            ax1.plot(mean_price, lw=2, color='blue', label='Moyenne')
            ax1.set_title('Projection des prix sur 30 jours')
            ax1.legend()
            
            # Volatilité
            for path in vol_paths:
                ax2.plot(path, lw=1, alpha=0.1, color='red')
            # Ajouter la moyenne des chemins
            mean_vol = np.mean(vol_paths, axis=0)
            ax2.plot(mean_vol, lw=2, color='red', label='Moyenne')
            ax2.set_title('Projection de la volatilité sur 30 jours')
            ax2.legend()
            
            st.pyplot(fig)
            
            # Afficher tableau avec valeurs de prédiction
            days = list(range(1, 31))
            forecast_df = pd.DataFrame({
                'Jour': days,
                'Prix moyen': mean_price,
                'Volatilité moyenne': mean_vol
            })
            
            st.subheader("Prévisions numériques")
            st.dataframe(forecast_df)
            
            # Statistiques
            st.subheader("Statistiques estimées")
            st.write(f"Moyenne estimée (μ): {mu:.6f}")
            st.write(f"Variance estimée (σ²): {sigma_sq:.6f}")
            st.write(f"Lambda estimé (λ): {lambda_:.4f}")
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