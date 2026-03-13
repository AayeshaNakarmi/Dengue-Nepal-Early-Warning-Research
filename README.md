# Towards an Early Warning System for Dengue in Nepal  
### A Deep Learning and Explainable AI (XAI) Approach

## Project Overview
Dengue fever is an increasing public health challenge in Nepal, with recurring outbreaks that peak after the monsoon season. Current surveillance mechanisms such as the Early Warning and Reporting System (EWARS) primarily rely on reactive case reporting, often identifying outbreaks only after transmission has already intensified.

This project aims to **design and develop** a data-driven dengue early warning framework for Nepal using **deep learning, machine learning, and traditional statistical approaches**. The system integrates meteorological, demographic, and epidemiological data and emphasizes **model interpretability through Explainable AI (XAI)** to support future public health decision-making.

> ⚠️ **Note:** This repository represents an **ongoing research and development effort**. Model implementation, evaluation, and analysis are currently in progress.

---

## Objectives
- To develop a **deep learning–based dengue forecasting framework** using LSTM architectures.
- To **experiment with and compare** traditional statistical models, machine learning models, and deep learning models.
- To **analyze the influence of meteorological and demographic variables** on dengue transmission using XAI techniques.
- To explore the feasibility of **early outbreak risk forecasting** at district level in Nepal.

---

## Methodological Approach (Planned)
The study follows a **tiered modeling framework**, which is being implemented incrementally:

### 1. Traditional Statistical Models (Baseline)
- Poisson and Negative Binomial regression
- Generalized Additive Models (GAMs)
- ARIMAX models for time-series analysis

### 2. Machine Learning Models (Intermediate)
- Random Forest
- XGBoost
- Feature importance analysis using SHAP

### 3. Deep Learning Models (Primary Focus)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)
- Attention-based LSTM (LSTM-ATT)

### 4. Explainable AI (XAI)
- SHAP and LIME for interpretability
- Attention visualization and partial dependence analysis

---

## Data Sources
The project uses **secondary datasets** from official sources:

- **Meteorological Data:**  
  Department of Hydrology and Meteorology (DHM), Nepal  
  *(temperature, rainfall, humidity, wind, radiation, soil parameters)*

- **Epidemiological Data:**  
  Early Warning and Reporting System (EWARS), EDCD, MoHP  
  *(weekly confirmed dengue cases)*

- **Demographic Data:**  
  Central Bureau of Statistics (CBS) Nepal & World Bank  
  *(population density, urbanization, age structure, growth rate)*

**Temporal Coverage:** 2019–2025  
**Spatial Resolution:** District level  
**Temporal Resolution:** Weekly  

---

## Tech Stack (Planned / In Use)
### Programming & Environment
- Python 3.10+
- Kaggle Notebook
- Git & GitHub

### Libraries & Frameworks
- Pandas, NumPy, SciPy
- Scikit-learn
- TensorFlow / Keras
- Statsmodels
- XGBoost
- SHAP, LIME
- Matplotlib, Seaborn, Plotly
- GeoPandas, Folium

---

## Evaluation Strategy (Planned)
- Time-aware train–validation–test splits
- Evaluation metrics:
  - RMSE, MAE
  - Pearson correlation coefficient
  - Sensitivity and specificity for outbreak detection
- Statistical comparison using the Diebold–Mariano test

---

## Repository Structure (Work in Progress)
