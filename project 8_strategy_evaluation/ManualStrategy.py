import datetime as dt
from util import get_data
import pandas as pd
import numpy as np
import marketsimcode as mk
import matplotlib.pyplot as plt
import matplotlib
import indicators as ind

class ManualStrategy(object):
    def __init__(self):
        self.buy_order = []
        self.sell_order = []

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        flag = 0
        sym = symbol
        dates = pd.date_range(sd, ed)
        df_price = get_data([sym], dates)

        if 'SPY' not in sym:
            df_price.drop('SPY', axis=1, inplace=True)
        df_price = df_price.fillna(method='ffill')
        df_price = df_price.fillna(method='bfill')
        df_price = df_price / df_price.iloc[0, :]

        df_order = pd.DataFrame(columns=['Order', 'Shares'], index=df_price.index)
         # Get indicator values to make trades
        df_sma, df_psma = ind.get_simple_moving_average(df_price, lookback=14)
        df_bollinger, df_upper, df_lower, rm, sd = ind.get_bb(df_price, lookback=14)
        df_momentum = ind.get_momentum(df_price, lookback=14)
        # df_volatility = ind.get_volatility(df_price, lookback=14)

        # Plot indicators for report
        figure, axis = plt.subplots()
        axis.plot(df_price, color="green", label="Price")
        axis.plot(df_sma, color="blue", label='SMA')
        axis.plot(df_psma, color="turquoise", label='P/SMA')
        axis.plot(df_upper, color="red", label='Upper Band')
        axis.plot(df_lower, color="orange", label='Lower Band')
        axis.plot(df_bollinger, color="chocolate", label='Bollinger B%')
        axis.plot(df_momentum, color="purple", label='Momentum')
        plt.axhline(y=0.0, color='dimgrey', linestyle='--')
        plt.legend()
        plt.title("Trading Indicators")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.grid()
        figure = matplotlib.pyplot.gcf()
        figure.set_size_inches(14.5, 5, forward=True)
        figure.savefig('Indicators', dpi=100)
        plt.clf()

        for index in range(df_price.shape[0]):
            i = df_price.index[index]
            if flag == 1:
                if df_bollinger.loc[i, sym] > 0.8 or df_psma.loc[i, sym] > 1.2 or df_momentum.loc[i, sym] > 0.2:
                    df_order.loc[i] = ['Short', 2000]
                    flag = -1
                    self.sell_order.append(df_order.index[index])
            elif flag == -1:
                if df_bollinger.loc[i, sym] < 0.2 or df_psma.loc[i, sym] < 0.7 or df_momentum.loc[i, sym] < -0.2:
                    df_order.loc[i] = ['Long', 2000]
                    flag = 1
                    self.buy_order.append(df_order.index[index])
            elif flag == 0:
                if df_bollinger.loc[i, sym] < 0.2 or df_psma.loc[i, sym] < 0.6 or df_momentum.loc[i, sym] < -0.2:
                    df_order.loc[i] = ['Long', 1000]
                    flag = 1
                    self.buy_order.append(df_order.index[index])
                elif df_bollinger.loc[i, sym] > 0.8 or df_psma.loc[i, sym] > 1.2 or df_momentum.loc[i, sym] > 0.2:
                    df_order.loc[i] = ['Short', 1000]
                    flag = -1
                    self.sell_order.append(df_order.index[index])

        df_index = df_order.index
        df_trades = pd.DataFrame(data=df_order, index=df_index, columns=['Order', 'Shares'])
        df_trades['Shares'] = np.where(df_trades['Order'] == 'Long', df_trades['Shares'], -1 * df_trades['Shares'])
        df_trades.fillna(0, inplace=True)
        df_trades = df_trades.loc[:, ['Shares']]
        df_trades.columns = [sym]
        return df_trades



    def benchmark_stategy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31)):
        sym = symbol
        dates = pd.date_range(sd, ed)
        df_prices_benchmark = get_data([sym], dates)
        df_trades_benchmark = pd.DataFrame(index=df_prices_benchmark.index)
        df_trades_benchmark[sym] = 0
        df_trades_benchmark.loc[df_trades_benchmark.index.min()] = 1000
        df_trades_benchmark.loc[df_trades_benchmark.index.max()] = -1000
        return df_trades_benchmark

def author():
    return "mlukacsko3"


def main():
    """In Sample"""
    is_start_date = dt.datetime(2008, 1, 1)
    is_end_date = dt.datetime(2009, 12, 31)
    symbol = 'JPM'
    date = pd.date_range(is_start_date, is_end_date)
    startval = 100000
    commission = 9.95
    impact = 0.005

    # Manual
    ms = ManualStrategy()
    df_trades  = ms.testPolicy(symbol, is_start_date, is_end_date, startval)
    is_manual_portvals  = mk.compute_portvals(df_trades, startval, commission, impact)
    is_cr, is_avg, is_sddr, is_sr = mk.compute_stats(is_manual_portvals)
    normalized_manual_portvals = is_manual_portvals/is_manual_portvals.iloc[0]
    is_cr_norm, is_avg_norm, is_sddr_norm, is_sr_norm = mk.compute_stats(is_manual_portvals)

    # Benchmark
    df_benchmark = ms.benchmark_stategy(symbol, is_start_date, is_end_date)
    is_benchmark_portvals = mk.compute_portvals(df_benchmark, startval, commission, impact)
    is_cr_bench, is_avg_bench, is_sddr_bench, is_sr_bench = mk.compute_stats(is_benchmark_portvals)
    normalized_benchmark_portvals = is_benchmark_portvals/is_benchmark_portvals.iloc[0]
    is_cr_bench_norm, is_avg_bench_norm, is_sddr_bench_norm, is_sr_bench_norm = mk.compute_stats(is_benchmark_portvals)

    # Plot manual vs benchmanrk for in sample
    long = ms.buy_order
    short = ms.sell_order
    figure,axis = plt.subplots()
    axis.set(xlabel='Date', ylabel="Normalized Portfolio Value", title="Manual vs Benchmark Strategy: In Sample")
    axis.plot(normalized_manual_portvals, color="red", label="Manual Strategy")
    axis.plot(normalized_benchmark_portvals, color="green", label='Benchmark')
    for date in long:
        axis.axvline(date, color="blue", label='Long')
    for date in short:
        axis.axvline(date, color="black", label='Short')
    handles, labels = axis.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    plt.legend(handle_list, label_list)
    plt.grid()
    figure = matplotlib.pyplot.gcf()
    figure.set_size_inches(14.5, 5, forward=True)
    figure.savefig('MS-InSample', dpi=100)
    plt.clf()

    """Out Of Sample"""
    startval = 100000
    commission = 9.95
    impact = 0.005
    symbol = "JPM"
    oos_start_date = dt.datetime(2010, 1, 1)
    oos_end_date = dt.datetime(2011, 12, 31)
    date = pd.date_range(oos_start_date, oos_end_date)

    # Manual
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol, oos_start_date, oos_end_date, startval)
    oos_manual_portvals = mk.compute_portvals(df_trades, startval, commission, impact)
    oos_cr, oos_avg, oos_sddr, oos_sr = mk.compute_stats(oos_manual_portvals)
    oos_normalized_manual_portvals = oos_manual_portvals / oos_manual_portvals.iloc[0]
    oos_cr_norm, oos_avg_norm, oos_sddr_norm, oos_sr_norm = mk.compute_stats(oos_normalized_manual_portvals)
    # Benchmark
    df_benchmark = ms.benchmark_stategy(symbol=symbol, sd=oos_start_date, ed=oos_end_date)
    oos_benchmark_portvals = mk.compute_portvals(df_benchmark, startval, commission, impact)
    oos_cr_bench, oos_avg_bench, oos_sddr_bench, oos_sr_bench = mk.compute_stats(oos_benchmark_portvals)
    oos_normalized_benchmark_portvals = oos_benchmark_portvals / oos_benchmark_portvals.iloc[0]
    oos_cr_bench_norm, oos_avg_bench_norm, oos_sddr_bench_norm, oos_sr_bench_norm = mk.compute_stats\
        (oos_normalized_benchmark_portvals)

    # Plot manual vs benchmark for out of sample
    buy_order = ms.buy_order
    sell_order = ms.sell_order
    figure, axis = plt.subplots()
    axis.plot(oos_normalized_manual_portvals, color="red", label="Manual Strategy")
    axis.plot(oos_normalized_benchmark_portvals, color="green", label='Benchmark')
    #oos_normalized_manual_portvals.plot(color="black", label="Manual Strategy")
    #oos_normalized_benchmark_portvals.plot(color="blue", label='Benchmark')
    for date in buy_order:
        axis.axvline(date, color="blue", label='Long')
    for date in sell_order:
        axis.axvline(date, color="black", label='Short')
    handles, labels = axis.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    plt.legend(handle_list, label_list)
    plt.title("Manual vs Benchmark Strategy: Out Of Sample")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Values")
    plt.grid()
    figure = matplotlib.pyplot.gcf()
    figure.set_size_inches(14.5, 5, forward=True)
    figure.savefig('MS-OutOfSample', dpi=100)
    plt.clf()

    # Write results to results text file
    file = open("p8_results_ManualStrategy.txt", "w")
    file.write("In Sample Dates: " + str(is_start_date) + " to " + str(is_end_date) + " for " + str(symbol) +
               "\n" +
               "\nManual Strategy - In Sample" +
               "\nCumulative Return: " + str(is_cr) +
               "\nStandard Deviation: " + str(is_sddr) +
               "\nAverage Daily Return: " + str(is_avg) +
               "\nSharpe Ratio: " + str(is_sr) +
               "\nFinal Portfolio Value: " + str(is_manual_portvals[-1]) +
               "\nNormalized Final Portfolio Value: " + str(normalized_manual_portvals[-1]) +
               "\n" +
               "\nBenchmark Strategy - In Sample" +
               "\nCumulative Return: " + str(is_cr_bench) +
               "\nStandard Deviation: " + str(is_sddr_bench) +
               "\nAverage Daily Return: " + str(is_avg_bench) +
               "\nSharpe Ratio: " + str(is_sr_bench) +
               "\nFinal Portfolio Value: " + str(is_benchmark_portvals[-1]) +
               "\nNormalized Final Portfolio Value: " + str(normalized_benchmark_portvals[-1]) +
               "\n" +
               "\nOut Of Sample Dates: " + str(oos_start_date) + " to " + str(oos_end_date) + " for " + str(symbol) +
               "\n" +
               "\nManual Strategy - Out Of Sample" +
               "\nCumulative Return: " + str(oos_cr) +
               "\nStandard Deviation: " + str(oos_sddr) +
               "\nAverage Daily Return: " + str(oos_avg) +
               "\nSharpe Ratio: " + str(oos_sr) +
               "\nFinal Portfolio Value: " + str(oos_manual_portvals[-1]) +
               "\nNormalized Final Portfolio Value: " + str(oos_normalized_manual_portvals[-1]) +
               "\n" +
               "\nBenchmark Strategy - Out Of Sample" +
               "\nCumulative Return: " + str(oos_cr_bench) +
               "\nStandard Deviation: " + str(oos_sddr_bench) +
               "\nAverage Daily Return: " + str(oos_avg_bench) +
               "\nSharpe Ratio: " + str(oos_sr_bench) +
               "\nFinal Portfolio Value: " + str(oos_benchmark_portvals[-1]) +
               "\nNormalized Final Portfolio Value: " + str(oos_normalized_benchmark_portvals[-1]))
    file.close()


if __name__ == "__main__":
    main()
