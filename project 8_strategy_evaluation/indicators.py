import matplotlib
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt


def get_simple_moving_average(df_prices, lookback):
    sma = df_prices.rolling(window=lookback,center=False).mean()
    psma = df_prices/sma
    return sma, psma


def get_bb(df_prices, lookback):
    sma_dontUse = df_prices.rolling(window=lookback).mean()
    sd = df_prices.rolling(window=lookback).std()
    topband = sma_dontUse + (2 * sd)
    lowerband = sma_dontUse - (2 * sd)
    bb = (df_prices - lowerband) / (topband - lowerband)
    return bb, topband, lowerband, sma_dontUse, sd


def get_momentum(df_prices, lookback):
    df_momentum = (df_prices / df_prices.shift(periods=lookback)) - 1
    return df_momentum


def get_volatility(df_prices, lookback):
    volatility = df_prices.rolling(window=lookback).std()
    return volatility


def author():
    return "mlukacsko3"


def test_code():
    lookback = 14
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    syms = ['JPM']
    dates = pd.date_range(sd, ed)
    df_prices = get_data(syms, dates)
    df_prices = df_prices[syms]
    normalized_prices = df_prices / df_prices.iloc[0, :]
    sma = get_simple_moving_average(df_prices, lookback)
    get_bb(sma, normalized_prices, df_prices, lookback)
    get_momentum(df_prices, normalized_prices, lookback)
    get_volatility(normalized_prices, df_prices, lookback)


if __name__ == "__main__":
    test_code()