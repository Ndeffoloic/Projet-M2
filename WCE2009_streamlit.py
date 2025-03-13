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
    
    return X[:T]

def estimate_parameters(returns):
    """Estime μ, σ² et λ selon les formules du document (Section 3)"""
    mu = np.mean(returns)
    sigma_sq = 2 * np.var(returns)
    
    # Calcul de l'autocorrélation lag-1
    rho1 = np.corrcoef(returns[:-1], returns[1:])[0,1]
    lambda_ = -np.log(max(rho1, 1e-6))
    
    return mu, sigma_sq, lambda_

def main():
    st.title("Prédiction de Prix et Volatilité")
    
    # Sélection des données
    data_source = st.selectbox("Source des données", ["Yahoo Finance", "Fichier Excel"])
    
    if data_source == "Yahoo Finance":
        ticker = st.text_input("Symbole Yahoo Finance (ex: AAPL)", "AAPL")
        data = yf.download(ticker, period="1y")
        returns = data['Close'].pct_change().dropna()
    else:
        uploaded_file = st.file_uploader("Télécharger fichier Excel", type=["xlsx"])
        if uploaded_file:
            data = pd.read_excel(uploaded_file)
            returns = data['Close'].pct_change().dropna()
    
    if 'returns' in locals():
        # Estimation des paramètres
        mu, sigma_sq, lambda_ = estimate_parameters(returns)
        
        # Paramètres de simulation
        st.sidebar.header("Paramètres de simulation")
        n_simulations = st.sidebar.number_input("Nombre de simulations", 100, 1000, 500)
        a = st.sidebar.number_input("Paramètre a (IG)", 0.1, 10.0, 2.2395e-7)
        b = st.sidebar.number_input("Paramètre b (IG)", 0.1, 10.0, 1.0)
        
        # Simulation de la volatilité
        vol_paths = np.zeros((n_simulations, 30))
        price_paths = np.zeros((n_simulations, 30))
        
        last_price = data['Close'].iloc[-1]
        
        for i in range(n_simulations):
            vol = simulate_ig_ou(X0=returns.std(), 
                                lambda_=lambda_, 
                                a=a, 
                                b=b)
            
            # Simulation des prix (Formule 4.1 adaptée)
            prices = [last_price]
            for t in range(29):
                drift = mu + (0.5 - 0.5)*vol[t]**2
                shock = vol[t] * np.random.normal()
                prices.append(prices[-1] * np.exp(drift + shock))
            
            vol_paths[i] = vol[:30]
            price_paths[i] = prices
        
        # Visualisation
        st.subheader("Résultats de simulation")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Prix
        for path in price_paths:
            ax1.plot(path, lw=1, alpha=0.1, color='blue')
        ax1.set_title('Projection des prix sur 30 jours')
        
        # Volatilité
        for path in vol_paths:
            ax2.plot(path, lw=1, alpha=0.1, color='red')
        ax2.set_title('Projection de la volatilité sur 30 jours')
        
        st.pyplot(fig)
        
        # Statistiques
        st.write(f"Moyenne estimée (μ): {mu:.6f}")
        st.write(f"Variance estimée (σ²): {sigma_sq:.6f}")
        st.write(f"Lambda estimé (λ): {lambda_:.4f}")

if __name__ == "__main__":
    main()