# 📈 IG-OU Asset Price Prediction - Implementation of WCE 2009 Paper

**Implementation of the paper:**  
*"Financial Modelling with Ornstein-Uhlenbeck Processes Driven by Lévy Process" by Ömer Önalan (WCE 2009)*  
[![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.23230.28487-blue)]()

## 🔍 Overview
Ce projet implémente le modèle IG-OU (Inverse Gaussian Ornstein-Uhlenbeck) décrit dans l'article WCE 2009 pour la prédiction des prix d'actifs financiers. L'application Streamlit permet de :

- Charger des données depuis Yahoo Finance ou des fichiers locaux
- Visualiser et analyser les données historiques
- Simuler les prix futurs avec le modèle IG-OU
- Comparer avec le modèle Black-Scholes classique
- Analyser la surface de volatilité

## 📚 Fondements Théoriques (Article WCE 2009)
### Composants clés implémentés :
1. **Processus IG-OU** (Section III)
   - Implémentation rigoureuse de l'équation 3.17
   - Algorithme optimisé de génération IG
   - Gestion robuste des cas limites

2. **Estimation des paramètres** (Section III.C)
   - Méthode des moments avec validation
   - Estimation robuste de λ par autocorrélation
   - Gestion des données manquantes et aberrantes

3. **Modélisation de la volatilité** 
   - Surface de volatilité dynamique
   - Clustering et mean-reversion
   - Comparaison avec Black-Scholes

## ⚙️ Structure du Projet
```bash
project/
├── app.py                  # Application principale Streamlit
├── core/                   # Cœur métier
│   ├── models/            # Modèles mathématiques
│   │   ├── ig_ou.py      # Modèle IG-OU (Eq. 3.17)
│   │   └── black_scholes.py
│   ├── estimators/       # Estimation des paramètres
│   │   └── parameters.py
│   └── data/            # Gestion des données
│       └── loader.py
├── ui/                   # Interface utilisateur
│   ├── components/      # Composants Streamlit
│   │   ├── sidebar.py
│   │   └── visualizations.py
│   └── helpers.py
└── tests/               # Tests unitaires et d'intégration
    ├── test_financial_models.py
    ├── test_data_handling.py
    ├── test_volatility_modeling.py
    ├── test_streamlit_ui.py
    └── test_performance.py
```

## 🚀 Installation

1. **Cloner le dépôt :**
```bash
git clone https://github.com/votre-username/votre-repo.git
cd votre-repo
```

2. **Créer un environnement virtuel :**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

## 🖥️ Utilisation

### Lancer l'application
```bash
streamlit run app.py
```

### Options disponibles
1. **Source des données :**
   - Yahoo Finance (temps réel)
   - Fichiers Excel/CSV
   - Données d'exemple préchargées

2. **Paramètres de simulation :**
   - Nombre de simulations
   - Horizon de prédiction
   - Choix du modèle (IG-OU, Black-Scholes, ou les deux)

3. **Visualisations :**
   - Données historiques et rendements
   - Surface de volatilité
   - Comparaison des modèles
   - Intervalles de confiance

## 🧪 Tests
Le projet inclut une suite de tests complète :

```bash
# Exécuter tous les tests
python -m pytest tests/

# Tests spécifiques
python -m pytest tests/test_financial_models.py
python -m pytest tests/test_performance.py
```

## 📊 Validation des Données
- Vérification automatique de la qualité des données
- Détection des valeurs aberrantes
- Gestion des données manquantes
- Validation de la fréquence d'échantillonnage

## 📌 Notes Importantes
1. **Performance :**
   - Optimisation vectorielle avec NumPy
   - Gestion efficace de la mémoire
   - Tests de performance intégrés

2. **Limitations :**
   - Les paramètres par défaut peuvent nécessiter un ajustement
   - La qualité des prédictions dépend des données d'entrée
   - Utilisation académique recommandée

## 📚 Références
- [1] Ö. Önalan (2009). *Financial Modelling with Ornstein-Uhlenbeck Processes...* WCE 2009
- [2] Barndorff-Nielsen & Shephard (2001). *Non-Gaussian OU-based models...*

## ⚠️ Disclaimer
Ce projet est une implémentation académique. Ne pas utiliser pour des décisions financières réelles sans validation approfondie.