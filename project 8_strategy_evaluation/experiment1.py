import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
from util import get_data
import matplotlib.pyplot as plt
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as mk

def main():
    startval = 100000
    commission = 9.95
    impact = 0.005
    symbol = "JPM"
    is_start_date = dt.datetime(2008, 1, 1)
    is_end_date = dt.datetime(2009, 12, 31)
    date = pd.date_range(is_start_date, is_end_date)

    manual = ms.ManualStrategy()
    strategy = sl.StrategyLearner()

    """Benchmark In Sample"""
    df_benchmark = manual.benchmark_stategy(symbol=symbol, sd=is_start_date, ed=is_end_date)
    is_benchmark_portvals = mk.compute_portvals(df_benchmark, startval, commission, impact)
    is_cr1, is_avg1, is_sddr1, is_sr1 = mk.compute_stats(is_benchmark_portvals)
    trades_total1 = df_benchmark[df_benchmark != 0].count()[0]

    """Manual In Sample"""
    df_trades = manual.testPolicy(symbol, is_start_date, is_end_date, startval)
    is_manual_portvals = mk.compute_portvals(df_trades, startval, commission, impact)
    is_cr2, is_avg2, is_sddr2, is_sr2 = mk.compute_stats(is_manual_portvals)
    trades_total2 = df_trades[df_trades != 0].count()[0]

    """Strategy Learner In Sample"""
    strategy.add_evidence(symbol, sd=is_start_date, ed=is_end_date, sv=startval)
    sl_df_trades = strategy.testPolicy(symbol, sd=is_start_date, ed=is_end_date, sv=startval)
    is_sl_portvals = mk.compute_portvals(sl_df_trades, startval, commission, impact)
    is_cr3, is_avg3, is_sddr3, is_sr3 = mk.compute_stats(is_sl_portvals)
    trades_total3 = sl_df_trades[sl_df_trades != 0].count()[0]

    # Normalize portvals
    normalized_benchmark_portvals = is_benchmark_portvals / is_benchmark_portvals.iloc[0]
    normalized_manual_portvals = is_manual_portvals / is_manual_portvals.iloc[0]
    normalized_sl_portvals = is_sl_portvals / is_sl_portvals.iloc[0]

    # Plot Results
    normalized_sl_portvals.plot(color='red', label='Strategy Learner')
    normalized_manual_portvals.plot(color='green', label='Manual Strategy')
    normalized_benchmark_portvals.plot(color='blue', label='Benchmark Strategy')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title("Experiment 1: Benchmark vs Manual Strategy vs Strategy Learner")
    plt.grid()
    plt.savefig('Experiment1.png')
    # plt.show()
    plt.clf()


    # Write results to text
    file = open("p8_results_Experiment1.txt", "w")
    file.write("Dates: " + str(is_start_date) + " to " + str(is_end_date) + " for " + str(symbol) +
               "\n" +
               "\nBenchmark Strategy" +
               "\nCumulative Return: " + str(is_cr1) +
               "\nStandard Deviation: " + str(is_sddr1) +
               "\nAverage Daily Return: " + str(is_avg1) +
               "\nSharpe Ratio: " + str(is_sr1) +
               "\nFinal Portfolio Value: " + str(is_benchmark_portvals[-1]) +
               "\nTotal Trades: " + str(trades_total1) +
               "\n" +
               "\nManual Strategy" +
               "\nCumulative Return: " + str(is_cr2) +
               "\nStandard Deviation: " + str(is_sddr2) +
               "\nAverage Daily Return: " + str(is_avg2) +
               "\nSharpe Ratio: " + str(is_sr2) +
               "\nFinal Portfolio Value: " + str(is_manual_portvals[-1]) +
               "\nTotal Trades: " + str(trades_total2) +
               "\n" +
               "\nStrategy Learner" +
               "\nCumulative Return: " + str(is_cr3) +
               "\nStandard Deviation: " + str(is_sddr3) +
               "\nAverage Daily Return: " + str(is_avg3) +
               "\nSharpe Ratio: " + str(is_sr3) +
               "\nFinal Portfolio Value: " + str(is_sl_portvals[-1]) +
               "\nTotal Trades: " + str(trades_total3) +
               "\n")
    file.close()

def author():
    return "mlukacsko3"


if __name__ == "__main__":
    np.random.seed(123456)
    main()
