import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm


# Simulation d'un processus OU pour les prix futurs
def simulate_OU_process(S0, mu, sigma, lambda_, days=30):
    dt = 1  # Intervalle de temps (1 jour)
    prices = [S0]
    for _ in range(days):
        dW = np.random.normal(0, np.sqrt(dt))  # Mouvement brownien
        dS = -lambda_ * (prices[-1] - mu) * dt + sigma * dW
        prices.append(prices[-1] + dS)
    return np.array(prices)

# Interface Streamlit
st.title("Prédiction de Prix et Volatilité d'un Actif")

# Choix de la source des données
data_source = st.radio("Source des données :", ("Yahoo Finance", "Fichier Excel"))

if data_source == "Yahoo Finance":
    ticker = st.text_input("Entrez le symbole de l'actif :", "AAPL")
    if st.button("Charger les données"):
        try:
            data = yf.download(ticker, period="1y")
            if not data.empty and "Close" in data.columns:
                st.line_chart(data["Close"])
                
                # Stockage des données dans une session state pour y accéder plus tard
                st.session_state.data = data
            else:
                st.error(f"Aucune donnée trouvée pour le symbole {ticker}")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement des données: {e}")
        
elif data_source == "Fichier Excel":
    uploaded_file = st.file_uploader("Chargez un fichier Excel", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file, thousands=',')  # Gestion des séparateurs de milliers
                
            # S'assurer que la colonne Close existe
            if "Close" in data.columns:
                # Convertir les valeurs de type string en float si nécessaire
                if data["Close"].dtype == object:
                    data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
                
                st.line_chart(data["Close"])
                
                # Stockage des données dans une session state
                st.session_state.data = data
            else:
                st.error("La colonne 'Close' est introuvable dans le fichier")
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier: {e}")

# Simulation et affichage des prévisions
if hasattr(st.session_state, 'data') and not st.session_state.data.empty:
    data = st.session_state.data
    
    # Vérifier que la colonne Close existe et n'est pas vide
    if "Close" in data.columns and not data["Close"].empty:
        S0 = data["Close"].iloc[-1]
        mu = data["Close"].mean()
        sigma = data["Close"].std()
        lambda_ = 0.1  # Valeur arbitraire, à calibrer
        
        prices = simulate_OU_process(S0, mu, sigma, lambda_)
        st.line_chart(pd.DataFrame(prices, columns=["Prix prédit"]))
    else:
        st.warning("Impossible de simuler: la colonne 'Close' est vide ou inexistante")