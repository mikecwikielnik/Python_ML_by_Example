"""
Predicting Stock Prices with Regression Algorithms

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 213). Packt Publishing. Kindle Edition.
"""

import pandas as pd
import yfinance as yf

djia_data = yf.download("^DJi", start="2005-12-01", end="2005-12-10", group_by='tickers')
djia_data

