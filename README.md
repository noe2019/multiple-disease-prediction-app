# Multiple Disease Prediction

Early-risk prediction for type 2 diabetes, heart disease and Parkinson's disease from clinical and voice biomarkers.

**Live app:** https://noe-multi-disease-5ba608b87c4d.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- Three models (diabetes, heart disease, Parkinson's)
- Model comparison with ROC-AUC on a hold-out set
- Interactive risk calculator per disease
- Feature-importance charts

## Tech stack
Python, scikit-learn, Streamlit, Plotly

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
