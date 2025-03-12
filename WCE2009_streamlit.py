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
        data = yf.download(ticker, period="1y")
        st.line_chart(data["Close"])
        
elif data_source == "Fichier Excel":
    uploaded_file = st.file_uploader("Chargez un fichier Excel", type=["xlsx", "csv"])
    if uploaded_file:
        data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
        st.line_chart(data["Close"])

# Simulation et affichage des prévisions
if 'data' in locals():
    S0 = data["Close"].iloc[-1]
    mu = data["Close"].mean()
    sigma = data["Close"].std()
    lambda_ = 0.1  # Valeur arbitraire, à calibrer
    
    prices = simulate_OU_process(S0, mu, sigma, lambda_)
    st.line_chart(pd.DataFrame(prices, columns=["Prix prédit"]))
