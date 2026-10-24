# 💎 Diamond Price Predictor

**Hamna Mubarak · FA24-BSE-046 · Section A** — Machine Learning Fundamentals (AIC354), Assignment 3

Streamlit web app that predicts a diamond's price using an **Extra Trees** model, chosen by comparing
5 algorithms (Ridge, Lasso, AdaBoost, Extra Trees, MLP) with **10-fold cross validation** on the full
diamonds dataset (53,775 unique diamonds). Cross-validation RMSE ≈ **$531**.

Live app: https://hamna-diamond-price.streamlit.app

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
