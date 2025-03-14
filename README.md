```markdown
# 📈 IG-OU Asset Price Prediction - Implementation of WCE 2009 Paper

**Implementation of the paper:**  
*"Financial Modelling with Ornstein-Uhlenbeck Processes Driven by Lévy Process" by Ömer Önalan (WCE 2009)*  
[![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.23230.28487-blue)]()

## 🔍 Overview
Ce projet implémente le modèle IG-OU (Inverse Gaussian Ornstein-Uhlenbeck) décrit dans l'article WCE 2009 pour la prédiction des prix d'actifs financiers. L'application Streamlit permet de :

- Sélectionner entre 2 actifs prédéfinis (BTC ou GLE.PA)
- Choisir un échelonnage temporel (minute/heure/jour/semaine/mois)
- Visualiser les simulations de prix et volatilité
- Comparer avec le modèle Black-Scholes classique

## 📚 Fondements Théoriques (Article WCE 2009)
### Composants clés implémentés :
1. **Processus IG-OU** (Section III)
   - `models.IGOUModel` : Implémente l'équation 3.17 de simulation
   - Algorithme de génération IG (Section 3.16) dans `IGOUModel.generate_ig()`

2. **Estimation des paramètres** (Section III.C)
   - Méthode des moments (Équations 3.15) dans `parameter_estimator.ParameterEstimator`
   - Autocorrélation exponentielle (Théorème p.3) pour l'estimation de λ

3. **Modélisation de la volatilité** 
   - Lien volume/volatilité (Équation 4.5) avec γ=2.2395e-7 (valeur par défaut dans l'UI)

### Ajustements par rapport à l'article :
- Utilisation de données prétraitées au lieu de l'API Yahoo Finance
- Implémentation modularisée suivant les principes SOLID
- Ajout d'une interface visuelle avec Streamlit

## ⚙️ Structure du Code
```bash
.
├── app.py               # Interface principale
├── data_loader.py       # Chargement des données Excel (Section IV.A)
├── models.py            # Implémentation IG-OU/Black-Scholes (Section III)
├── parameter_estimator.py # Estimation μ, σ², λ (Section III.C)
└── assets/              # Données prétraitées par échelonnage
    ├── minute/
    │   ├── BTC.csv
    │   └── GLE.PA.csv
    └── .../
```

## 🚀 Installation
1. **Environnement virtuel :**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. **Dépendances :**
```bash
pip install -r requirements.txt
```

Contenu de `requirements.txt` :
```
streamlit==1.22.0
numpy==1.23.5
pandas==1.5.3
matplotlib==3.7.0
scipy==1.10.0
```

3. **Lancer l'application :**
```bash
streamlit run app.py
```

## 🖥️ Usage
1. Sélectionner un actif et un échelonnage
2. Ajuster les paramètres de simulation dans la sidebar
3. Visualiser les prédictions interactives
4. Comparer les modèles IG-OU vs Black-Scholes

## 📌 Notes importantes
- Les données doivent être structurées comme suit :
  ```csv
  Date,Close
  2023-01-01 00:00:00,42000.0
  2023-01-01 00:01:00,42012.5
  ...
  ```
- La valeur par défaut du paramètre `a=2.2395e-7` provient de l'étude sur General Motors (Section IV.B)

## 📚 Références
- [1] Ö. Önalan (2009). *Financial Modelling with Ornstein-Uhlenbeck Processes...* WCE 2009
- [2] Barndorff-Nielsen & Shephard (2001). *Non-Gaussian OU-based models...*

## ⚠️ Disclaimer
Ce projet est une implémentation académique. Ne pas utiliser pour des décisions financières réelles. Les paramètres par défaut peuvent nécessiter un réétalonnage pour des actifs spécifiques.
```