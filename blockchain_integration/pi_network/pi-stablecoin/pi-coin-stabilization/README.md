Pi Coin Stabilization System
==========================

This system uses AI-powered stabilization to maintain the value of Pi Coin. It integrates multiple machine learning models to predict the future value of Pi Coin and adjust the stabilization mechanism accordingly.

Directory Structure
--------------------

* `ai_models`: Contains the machine learning models used for prediction.
* `models`: Contains the stabilization model that integrates the machine learning models.
* `predictors`: Contains the price and sentiment predictors.
* `utils`: Contains utility functions for data loading and feature engineering.
* `main.py`: The main script that loads data, engineers features, and predicts price and sentiment.

Requirements
------------

* `pandas`
* `scikit-learn`
* `keras`
* `statsmodels`
* `nltk`

Usage
-----

1. Install the required packages using `pip install -r requirements.txt`.
2. Run the `main.py` script using `python main.py`.
