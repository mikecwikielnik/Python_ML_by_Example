"""
Predicting Stock Prices with Regression Algorithms

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 213). Packt Publishing. Kindle Edition.
"""

import pandas as pd
import yfinance as yf

djia_data = yf.download("^DJi", start="2005-12-01", end="2005-12-10", group_by='tickers')
djia_data

def generate_features(df):
    """
    Generate features for a stock/index based on historical price and performance
    @param df: dataframe with columns "open", "close", "high", "low", "volume", "adjusted close"
    @return: dataframe, data set with new features
    """
    df_new = pd.DataFrame()
    # 6 original features
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)

    