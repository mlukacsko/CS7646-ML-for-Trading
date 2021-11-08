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
    df_prices = df_prices[symbol]
    df_prices['Cash'] = pd.Series(1.0, index=df_prices.index)

    row = df_prices.shape[0]
    column = df_prices.shape[1]
    df_trades = get_data(symbol, dates)
    df_trades = df_trades[symbol]
    df_trades['Cash'] = pd.Series(1.0, index=df_trades.index)
    df_trades.iloc[:, :] = np.zeros((row, column))

    for index, row in orders.iterrows():
        date = index
        sym = 'JPM'
        shares = row['Holdings']
        total = df_trades[sym].sum()
        if shares == 1000:
            if total == -1000:
                df_trades[sym][date] += 2000
                df_trades['Cash'][date] -= (df_prices[sym][date] * 2000)
            elif total == 0:
                df_trades[sym][date] += 1000
                df_trades['Cash'][date] -= (df_prices[sym][date] * 1000)
            elif total == 1000:
                df_trades[sym][date] += 0
                df_trades['Cash'][date] -= (df_prices[sym][date] * 0)
        elif shares == -1000:
            if total == -1000:
                df_trades[sym][date] -= 0
                df_trades['Cash'][date] += (df_prices[sym][date] * 0)
            elif total == 0:
                df_trades[sym][date] -= 1000
                df_trades['Cash'][date] += (df_prices[sym][date] * 1000)
            elif total == 1000:
                df_trades[sym][date] -= 2000
                df_trades['Cash'][date] += (df_prices[sym][date] * 2000)

    df_holdings = df_trades.copy()
    df_holdings['Cash'][start_date] += start_val
    df_holdings = df_holdings.cumsum()
    df_value = df_prices * df_holdings
    df_port_val = df_value.sum(axis=1)

    # Calculate daily return, cumulative return, avg daily return, and std daily return
    daily_rets = df_port_val.copy()
    daily_rets[1:] = (df_port_val[1:] / df_port_val[:-1].values) - 1
    daily_rets.iloc[0] = 0
    daily_rets = daily_rets[1:]
    cr = (df_port_val[-1] / df_port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    return df_port_val, cr, adr, sddr

def author():
    return "mlukacsko3"