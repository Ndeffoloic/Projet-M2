from typing import List, Optional

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


class VolatilityPlotter:
    """Génère les visualisations de volatilité selon les figures du document"""
    
    @staticmethod
    def plot_comparison(ig_paths: np.ndarray, bs_paths: np.ndarray, title: str = "Comparaison des prédictions de prix"):
        """
        Compare les trajectoires IG-OU et Black-Scholes
        
        Args:
            ig_paths: Trajectoires du modèle IG-OU (n_simulations x T)
            bs_paths: Trajectoires du modèle Black-Scholes (n_simulations x T)
            title: Titre du graphique
        """
        # Calcul des statistiques
        mean_ig = np.mean(ig_paths, axis=0)
        p05_ig = np.percentile(ig_paths, 5, axis=0)
        p95_ig = np.percentile(ig_paths, 95, axis=0)
        
        mean_bs = np.mean(bs_paths, axis=0)
        p05_bs = np.percentile(bs_paths, 5, axis=0)
        p95_bs = np.percentile(bs_paths, 95, axis=0)
        
        # Création du DataFrame pour Altair
        time_points = np.arange(len(mean_ig))
        df = pd.DataFrame({
            'Time': np.concatenate([time_points, time_points, time_points, time_points]),
            'Value': np.concatenate([mean_ig, p05_ig, p95_ig, mean_bs]),
            'Type': ['IG-OU (Moyenne)'] * len(mean_ig) + 
                   ['IG-OU (5%)'] * len(p05_ig) + 
                   ['IG-OU (95%)'] * len(p95_ig) +
                   ['Black-Scholes (Moyenne)'] * len(mean_bs)
        })
        
        # Création du graphique Altair
        base = alt.Chart(df).encode(
            x='Time:Q',
            y='Value:Q',
            color='Type:N'
        )
        
        # Lignes pour les moyennes
        lines = base.mark_line().encode(
            strokeDash=alt.condition(
                alt.datum.Type == 'Black-Scholes (Moyenne)',
                alt.value([5, 5]),
                alt.value([0, 0])
            )
        )
        
        # Zones pour les intervalles de confiance
        areas = base.mark_area(opacity=0.1).encode(
            y='Value:Q',
            y2='Value2:Q'
        ).transform_filter(
            alt.datum.Type != 'Black-Scholes (Moyenne)'
        )
        
        # Combinaison des graphiques
        chart = (lines + areas).properties(
            title=title,
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    
    @staticmethod
    def plot_volatility_surface(volatilities: np.ndarray, times: np.ndarray, title: str = "Surface de Volatilité"):
        """
        Génère une surface 3D de volatilité avec Plotly
        """
        X, Y = np.meshgrid(times, range(volatilities.shape[0]))
        
        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=volatilities,
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Temps',
                yaxis_title='Simulation',
                zaxis_title='Volatilité'
            ),
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def plot_historical_data(prices: pd.Series, returns: pd.Series):
        """
        Affiche les données historiques et les rendements avec Altair
        """
        # Création des DataFrames pour Altair
        prices_df = pd.DataFrame({
            'Date': prices.index,
            'Prix': prices.values
        })
        
        returns_df = pd.DataFrame({
            'Date': returns.index,
            'Rendement': returns.values
        })
        
        # Graphique des prix
        price_chart = alt.Chart(prices_df).mark_line().encode(
            x='Date:T',
            y='Prix:Q'
        ).properties(
            title='Prix Historiques',
            width=800,
            height=300
        ).interactive()
        
        # Graphique des rendements
        returns_chart = alt.Chart(returns_df).mark_line().encode(
            x='Date:T',
            y='Rendement:Q'
        ).properties(
            title='Rendements',
            width=800,
            height=300
        ).interactive()
        
        st.altair_chart(price_chart, use_container_width=True)
        st.altair_chart(returns_chart, use_container_width=True)
    
    @staticmethod
    def plot_diagnostics(returns, model_returns, vol_series, model_name="BNS"):
        """
        Génère des graphiques de diagnostic avec Altair
        """
        # Nettoyage des données
        def clean_data(data):
            if data is None:
                return None
            try:
                np_array = np.array(data, dtype=float)
                flattened = np_array.ravel()
                finite_mask = np.isfinite(flattened)
                
                if not np.any(finite_mask):
                    return None
                    
                cleaned = pd.Series(flattened[finite_mask])
                return cleaned if not cleaned.empty else None
                
            except Exception as e:
                st.error(f"Erreur de nettoyage: {str(e)}")
                return None
        
        returns = clean_data(returns)
        model_returns = clean_data(model_returns)
        
        if vol_series is not None:
            vol_series = clean_data(vol_series)
        
        # Création des colonnes pour l'affichage
        col1, col2 = st.columns(2)
        
        # Distribution des rendements
        if returns is not None and model_returns is not None:
            with col1:
                st.subheader('Distribution des rendements')
                df = pd.DataFrame({
                    'Rendement': np.concatenate([returns, model_returns]),
                    'Type': ['Données réelles'] * len(returns) + [f'Modèle {model_name}'] * len(model_returns)
                })
                
                chart = alt.Chart(df).mark_bar(opacity=0.5).encode(
                    x=alt.X('Rendement:Q', bin=alt.Bin(maxbins=30)),
                    y='count()',
                    color='Type:N'
                ).properties(
                    width=400,
                    height=300
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
        
        # Q-Q Plot
        if returns is not None:
            with col2:
                st.subheader('Q-Q Plot')
                qq_data = pd.DataFrame({
                    'Quantiles théoriques': stats.norm.ppf(np.linspace(0.01, 0.99, len(returns))),
                    'Quantiles observés': np.sort(returns)
                })
                
                chart = alt.Chart(qq_data).mark_point().encode(
                    x='Quantiles théoriques:Q',
                    y='Quantiles observés:Q'
                ).properties(
                    width=400,
                    height=300
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
        
        # ACF des résidus
        if returns is not None and model_returns is not None:
            st.subheader('ACF des résidus carrés')
            residuals = returns - model_returns
            squared_residuals = residuals ** 2
            
            from statsmodels.graphics.tsaplots import acf
            acf_values = acf(squared_residuals, nlags=40)
            
            df = pd.DataFrame({
                'Lag': range(len(acf_values)),
                'ACF': acf_values
            })
            
            chart = alt.Chart(df).mark_bar().encode(
                x='Lag:Q',
                y='ACF:Q'
            ).properties(
                width=800,
                height=300
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)

    @staticmethod
    def plot_autocorrelation_comparison(actual_returns, igou_returns, bs_returns, bns_returns, lags=40):
        """Figure 2: True vs Estimated Autocorrelation avec Altair"""
        # S'assurer que les séries sont des pd.Series
        def ensure_series(data):
            if not isinstance(data, pd.Series):
                return pd.Series(data)
            return data
            
        actual_returns = ensure_series(actual_returns)
        igou_returns = ensure_series(igou_returns)
        bs_returns = ensure_series(bs_returns)
        bns_returns = ensure_series(bns_returns)
        
        # Limiter lags à la moitié de la longueur des données minimale
        min_length = min(len(actual_returns), len(igou_returns), len(bs_returns), len(bns_returns))
        max_lags = min(lags, min_length // 2)
        
        # Calcul des autocorrélations
        from statsmodels.tsa.stattools import acf
        
        try:
            acf_actual = acf(actual_returns, nlags=max_lags, fft=True)
            acf_igou = acf(igou_returns, nlags=max_lags, fft=True)
            acf_bs = acf(bs_returns, nlags=max_lags, fft=True)
            acf_bns = acf(bns_returns, nlags=max_lags, fft=True)
            
            # Créer le DataFrame pour Altair
            df = pd.DataFrame({
                'Lag': np.concatenate([range(len(acf_actual))] * 4),
                'ACF': np.concatenate([acf_actual, acf_igou, acf_bs, acf_bns]),
                'Type': ['Données réelles'] * len(acf_actual) + 
                       ['IG-OU'] * len(acf_igou) + 
                       ['Black-Scholes'] * len(acf_bs) + 
                       ['BNS'] * len(acf_bns)
            })
            
            # Créer le graphique Altair
            base = alt.Chart(df).encode(
                x='Lag:Q',
                y='ACF:Q',
                color='Type:N'
            )
            
            # Points pour les données réelles
            points = base.mark_point(
                filled=True,
                size=60
            ).transform_filter(
                alt.datum.Type == 'Données réelles'
            )
            
            # Lignes pour les modèles avec des styles différents
            lines = base.mark_line().encode(
                strokeDash=alt.StrokeDash(
                    field='Type',
                    scale=alt.Scale(
                        domain=['IG-OU', 'Black-Scholes', 'BNS'],
                        range=[[0, 0], [5, 5], [2, 2]]  # Styles de ligne différents pour chaque modèle
                    )
                )
            ).transform_filter(
                alt.datum.Type != 'Données réelles'
            )
            
            # Combiner les graphiques
            chart = (points + lines).properties(
                title='Autocorrélation réelle vs estimée',
                width=800,
                height=400
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors du calcul des autocorrélations: {str(e)}")

    @staticmethod
    def plot_returns_comparison(actual_returns, igou_returns, bs_returns, bns_returns):
        """Figure 6: Historical vs Estimated Returns avec Altair"""
        # Convertir en numpy arrays si nécessaire
        actual_returns = np.array(actual_returns)
        igou_returns = np.array(igou_returns)
        bs_returns = np.array(bs_returns)
        bns_returns = np.array(bns_returns)
        
        # Déterminer la longueur minimale
        min_length = min(len(actual_returns), len(igou_returns), len(bs_returns), len(bns_returns))
        
        # Créer le DataFrame pour Altair
        df = pd.DataFrame({
            'Time': np.concatenate([range(min_length)] * 4),
            'Returns': np.concatenate([
                actual_returns[:min_length],
                igou_returns[:min_length],
                bs_returns[:min_length],
                bns_returns[:min_length]
            ]),
            'Type': ['Rendements réels'] * min_length + 
                   ['IG-OU'] * min_length + 
                   ['Black-Scholes'] * min_length + 
                   ['BNS'] * min_length
        })
        
        # Créer le graphique Altair
        chart = alt.Chart(df).mark_line().encode(
            x='Time:Q',
            y='Returns:Q',
            color='Type:N',
            strokeDash=alt.condition(
                alt.datum.Type == 'Black-Scholes',
                alt.value([5, 5]),
                alt.value([0, 0])
            )
        ).properties(
            title='Comparaison des rendements historiques et estimés',
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    @staticmethod
    def plot_residuals_acf(actual_returns, model_returns, model_name):
        """Figure 8: ACF of Squared Residuals avec Altair"""
        # Convertir en séries si nécessaire
        if not isinstance(actual_returns, pd.Series):
            actual_returns = pd.Series(actual_returns)
        if not isinstance(model_returns, pd.Series):
            model_returns = pd.Series(model_returns)
            
        try:
            # Réinitialiser les index
            actual_returns = actual_returns.reset_index(drop=True)
            model_returns = model_returns.reset_index(drop=True)
            
            # Calculer les résidus
            min_length = min(len(actual_returns), len(model_returns))
            residuals = actual_returns.values[:min_length] - model_returns.values[:min_length]
            squared_residuals = residuals ** 2
            
            # Calculer l'ACF
            from statsmodels.tsa.stattools import acf
            acf_values = acf(squared_residuals, nlags=min(40, min_length // 2))
            
            # Créer le DataFrame pour Altair
            df = pd.DataFrame({
                'Lag': range(len(acf_values)),
                'ACF': acf_values
            })
            
            # Créer le graphique Altair
            chart = alt.Chart(df).mark_bar().encode(
                x='Lag:Q',
                y='ACF:Q',
                tooltip=['Lag:Q', 'ACF:Q']
            ).properties(
                title=f'ACF des résidus carrés ({model_name})',
                width=800,
                height=400
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors du calcul de l'ACF des résidus: {str(e)}")

def plot_predictions(price_paths: List[List[float]], bs_prices: List[float]):
    """Plot price predictions with Altair"""
    # Convertir en numpy arrays
    price_paths = np.array(price_paths)
    bs_prices = np.array(bs_prices)
    
    # Calculer les statistiques
    mean_path = np.mean(price_paths, axis=0)
    std_path = np.std(price_paths, axis=0)
    upper_bound = mean_path + 2 * std_path
    lower_bound = mean_path - 2 * std_path
    
    # Créer le DataFrame pour Altair
    time_points = np.arange(len(mean_path))
    df = pd.DataFrame({
        'Time': np.concatenate([time_points] * 3),
        'Price': np.concatenate([mean_path, upper_bound, lower_bound]),
        'Type': ['Moyenne IG-OU'] * len(mean_path) + 
               ['Limite supérieure'] * len(upper_bound) + 
               ['Limite inférieure'] * len(lower_bound)
    })
    
    # Créer le graphique Altair
    base = alt.Chart(df).encode(
        x='Time:Q',
        y='Price:Q',
        color='Type:N'
    )
    
    # Ligne pour la moyenne
    mean_line = base.mark_line().transform_filter(
        alt.datum.Type == 'Moyenne IG-OU'
    )
    
    # Zone pour les intervalles de confiance
    confidence_band = base.mark_area(opacity=0.2).transform_filter(
        alt.datum.Type != 'Moyenne IG-OU'
    )
    
    # Ligne pour Black-Scholes
    bs_df = pd.DataFrame({
        'Time': time_points,
        'Price': bs_prices,
        'Type': 'Black-Scholes'
    })
    
    bs_line = alt.Chart(bs_df).mark_line(
        strokeDash=[5, 5]
    ).encode(
        x='Time:Q',
        y='Price:Q',
        color='Type:N'
    )
    
    # Combiner les graphiques
    chart = (mean_line + confidence_band + bs_line).properties(
        title='Prédictions de prix',
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

def plot_volatility(vol_paths: List[List[float]]):
    """Plot volatility paths with Altair"""
    # Convertir en numpy array
    vol_paths = np.array(vol_paths)
    
    # Calculer les statistiques
    mean_vol = np.mean(vol_paths, axis=0)
    std_vol = np.std(vol_paths, axis=0)
    upper_bound = mean_vol + 2 * std_vol
    lower_bound = mean_vol - 2 * std_vol
    
    # Créer le DataFrame pour Altair
    time_points = np.arange(len(mean_vol))
    df = pd.DataFrame({
        'Time': np.concatenate([time_points] * 3),
        'Volatility': np.concatenate([mean_vol, upper_bound, lower_bound]),
        'Type': ['Moyenne'] * len(mean_vol) + 
               ['Limite supérieure'] * len(upper_bound) + 
               ['Limite inférieure'] * len(lower_bound)
    })
    
    # Créer le graphique Altair
    base = alt.Chart(df).encode(
        x='Time:Q',
        y='Volatility:Q',
        color='Type:N'
    )
    
    # Ligne pour la moyenne
    mean_line = base.mark_line().transform_filter(
        alt.datum.Type == 'Moyenne'
    )
    
    # Zone pour les intervalles de confiance
    confidence_band = base.mark_area(opacity=0.2).transform_filter(
        alt.datum.Type != 'Moyenne'
    )
    
    # Combiner les graphiques
    chart = (mean_line + confidence_band).properties(
        title='Trajectoires de volatilité',
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
