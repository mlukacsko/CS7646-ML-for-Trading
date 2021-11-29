import pandas as pd
import numpy as np
from util import get_data

def compute_portvals(orders, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    orders.sort_index(ascending=True, inplace=True)
    start_date = orders.index.min()
    end_date = orders.index.max()
    dates = pd.date_range(start_date, end_date)

    symbol = np.array(['JPM'])

    df_prices = get_data(symbol, dates)
    df_prices.ffill(inplace=True)  # Forward Filling first
    df_prices.bfill(inplace=True)  # Backward filling
    df_prices = df_prices[symbol]
    df_prices['Cash'] = pd.Series(1.0, index=df_prices.index)

    rows = df_prices.shape[0]
    columns = df_prices.shape[1]
    df_trades = get_data(symbol, dates)
    df_trades = df_trades[symbol]
    df_trades['Cash'] = pd.Series(1.0, index=df_trades.index)
    df_trades.iloc[:, :] = np.zeros((rows, columns))
    df_trades.iloc[0, -1] = start_val

    for index, row in orders.iterrows():
        date = index
        sym = 'JPM'
        shares = row[sym]
        share_price = df_prices.loc[date, symbol]

        if shares < 0:
            df_trades.loc[date, symbol] = df_trades.loc[date, symbol] + shares
            share_price = share_price - (share_price * impact)
        else:
            df_trades.loc[date, symbol] = df_trades.loc[date, symbol] + shares
            share_price = share_price + (share_price * impact)
        cost = df_trades.loc[date, 'Cash'] - commission - (share_price * shares)
        df_trades['Cash'][date] = cost

    df_holdings = df_trades.copy()
    df_holdings = df_holdings.cumsum()
    df_value = df_prices * df_holdings
    df_port_val = df_value.sum(axis=1)
    return df_port_val

def compute_stats(df_port_val):
    # Calculate daily return, cumulative return, avg daily return, and std daily return
    daily_rets = df_port_val.copy()
    daily_rets[1:] = (df_port_val[1:] / df_port_val[:-1].values) - 1
    daily_rets.iloc[0] = 0
    daily_rets = daily_rets[1:]
    cr = (df_port_val[-1] / df_port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(252) * (daily_rets.mean() / daily_rets.std())
    return cr, adr, sddr, sr

def author():
    return "mlukacsko3"