import datetime as dt
from util import get_data
import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

def testPolicy(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
    dates = pd.date_range(sd, ed)
    df_prices = get_data([symbol],dates)
    df_prices = df_prices[symbol]

    adjusted_price = pd.Series(np.nan, index=df_prices.index)
    adjusted_price[:-1] = df_prices[:-1] / df_prices.values[1:] - 1
    order_sign = (-1) * adjusted_price.apply(np.sign)
    purchase = order_sign.diff() / 2
    purchase[0] = order_sign[0]


    trade_list = []
    for date in purchase.index:
        if purchase.loc[date] == 1:
            trade_list.append((date, 1000))
        elif purchase.loc[date] == -1:
            trade_list.append((date, -1000))
        elif purchase.loc[date] == 0:
            trade_list.append((date, 0))

    df_trades = pd.DataFrame(trade_list, columns=["Dates", "Holdings"])
    df_trades.set_index("Dates", append=False, inplace=True)

    return df_trades

def author():
    return 'mlukacsko3'

def testCode():
    startval = 100000
    symbol = np.array(["JPM"])
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    date = pd.date_range(start_date, end_date)

    df_benchmark_value = get_data(symbol, date)
    df_benchmark_value = df_benchmark_value[symbol]

    df_benchmark_trades = np.zeros_like(df_benchmark_value)
    df_benchmark_trades[0] = 1000
    df_index = df_benchmark_value.index
    df_benchmark_trades = pd.DataFrame(data=df_benchmark_trades, index=df_index, columns=['Holdings'])

    benchmark_portvals, benchmark_cr, benchmark_addr, benchmark_sddr = compute_portvals(df_benchmark_trades, startval,
                                                                                        0.0, 0.0)

    df_trades = testPolicy()

    TOS_portvals, TOS_cr, TOS_addr, TOS_sddr = compute_portvals(df_trades, startval, 0.0, 0.0)

    tos_ending_value = TOS_portvals[-1]
    benchmark_ending_value = benchmark_portvals[-1]
    normalized_TOS = TOS_portvals / TOS_portvals.iloc[0]
    normalized_benchmark = benchmark_portvals / benchmark_portvals.iloc[0]

    file = open("p6_results.txt", "w")
    file.write("Benchmark Strategy Returns:" +
               "\nBenchmark Strategy Cumulative Return: " + str(benchmark_cr) +
               "\nBenchmark Strategy Avd Daily Return: " + str(benchmark_addr) +
               "\nBenchmark Strategy Std Daily Return: " + str(benchmark_sddr) +
               "\nBenchmark Strategy Normalized Ending Value: " + str(normalized_benchmark[-1]) +
               "\nBenchmark Strategy Ending Value: " + str(benchmark_ending_value) +
               "\n" +
               "\nOptimal Strategy Returns:" +
               "\nOptimal Strategy Cumulative Return: " + str(TOS_cr) +
               "\nOptimal Strategy Avd Daily Return: " + str(TOS_addr) +
               "\nOptimal Strategy Std Daily Return: " + str(TOS_sddr) +
               "\nOptimal Strategy Normalized Ending Value: " + str(normalized_TOS[-1]) +
               "\nOptimal Strategy Ending Value: " + str(tos_ending_value))
    file.close()

    plt.xticks(rotation=12)
    plt.title("Theoretically Optimal Strategy(TOS) vs. Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.plot(normalized_TOS, 'r', label="TOS Strategy")
    plt.plot(normalized_benchmark, 'g', label="Benchmark Strategy")
    plt.grid(True)
    plt.legend()
    plt.savefig("TOSvsBenchmark.png")
    plt.clf()

if __name__ == "__main__":
    testCode()

