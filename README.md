# Football Player Market Value Prediction

Final project for **CSC14119 – Introduction to Data Science**. An end-to-end data science pipeline that scrapes football player statistics from [Transfermarkt](https://www.transfermarkt.com/) / FBref-style sources, cleans and engineers features from them, trains multiple ML models to predict player market value, and ships the result as an interactive web app.

## Pipeline Overview

```
source/
├── Data Collection/       Scrapes raw player stats for 29 leagues → data/*.csv
├── Data Pre-processing/   Cleans & merges raw CSVs into one modeling-ready dataset
├── Data Exploration/      EDA + answers to meaningful analysis questions
└── Data Modeling/         Feature engineering, model training, and Streamlit app
```

Each folder has its own `README.md` and `requirements.txt` with setup/run instructions specific to that stage.

## Dataset

- **14,977** players scraped across **29** leagues worldwide, cleaned down to **14,823** rows.
- **62** engineered features used for modeling: per-90 performance stats (goals, assists, xG/xA, progressive passes/carries, SCA/GCA, defensive actions), log transforms, interaction and polynomial features, and K-Fold target-encoded categoricals (nationality, club) to avoid leakage.

## Meaningful Questions Explored

See [`Data Exploration/MeaningfulQuestion.ipynb`](source/Data%20Exploration/MeaningfulQuestion.ipynb) for the full analysis, including:
- Which playing position commands the highest market value?
- What is a player's "prime age" in terms of market value?
- How does finishing ability affect a forward's market value?

## Modeling

Three models were trained, tuned with `GridSearchCV`, and validated with 5-fold cross-validation:

| Model         | Test R² | RMSE (€M) | MAE (€M) | MAPE (%) |
|---------------|:-------:|:---------:|:--------:|:--------:|
| Random Forest | 0.7118  | 4.17      | 1.60     | 83.31    |
| **XGBoost**   | **0.7716** | **3.38** | **1.32** | **72.65** |
| LightGBM      | 0.7690  | 3.33      | 1.31     | 71.27    |

**XGBoost** was selected as the best model (explains ~77% of variance on held-out test data), with LightGBM close behind and roughly 5x faster to tune.

Full methodology (feature selection, baseline vs. tuned comparisons, overfitting checks) is in [`Data Modeling/DataModeling.ipynb`](source/Data%20Modeling/DataModeling.ipynb).

## Demo App

A Streamlit app lets you try the trained models interactively: upload a CSV for batch predictions, pick a player from the database, enter stats manually, or compare all three models side by side.

```bash
cd "source/Data Modeling"
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, Streamlit, Selenium (scraping).

## Team

Group project for CSC14119, University of Science, VNU-HCM.
