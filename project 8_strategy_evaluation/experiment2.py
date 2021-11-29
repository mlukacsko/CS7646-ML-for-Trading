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
    commission = 0.00
    sym = "JPM"
    is_start_date = dt.datetime(2008, 1, 1)
    is_end_date = dt.datetime(2009, 12, 31)

    # Strategy Learner 1: Impact = 0.000
    learner1 = sl.StrategyLearner(verbose=False, impact=0.0)
    learner1.add_evidence(symbol=sym, sd=is_start_date, ed=is_end_date)
    df_trades1 = learner1.testPolicy(symbol=sym, sd=is_start_date, ed=is_end_date, sv=startval)
    port_val1 = mk.compute_portvals(df_trades1, start_val=startval, commission=commission, impact=0.0)
    is_cr1, is_avg1, is_sddr1, is_sr1 = mk.compute_stats(port_val1)
    trades_total1 = df_trades1[df_trades1 != 0].count()[0]
    #print("sl1 done")

    # Strategy Learner 2: Impact = 0.025
    learner2 = sl.StrategyLearner(verbose=False, impact=0.025)
    learner2.add_evidence(symbol=sym, sd=is_start_date, ed=is_end_date)
    df_trades2 = learner2.testPolicy(symbol=sym, sd=is_start_date, ed=is_end_date, sv=startval)
    port_val2 = mk.compute_portvals(df_trades2, start_val=startval, commission=commission, impact=0.025)
    is_cr2, is_avg2, is_sddr2, is_sr2 = mk.compute_stats(port_val2)
    trades_total2 = df_trades2[df_trades2 != 0].count()[0]
    #print("sl2 done")

    # Strategy Learner 3: Impact = 0.050
    learner3 = sl.StrategyLearner(verbose=False, impact=0.050)
    learner3.add_evidence(symbol=sym, sd=is_start_date, ed=is_end_date)
    df_trades3 = learner3.testPolicy(symbol=sym, sd=is_start_date, ed=is_end_date, sv=startval)
    port_val3 = mk.compute_portvals(df_trades3, start_val=startval, commission=commission, impact=0.050)
    is_cr3, is_avg3, is_sddr3, is_sr3 = mk.compute_stats(port_val3)
    trades_total3 = df_trades3[df_trades3 != 0].count()[0]
    #print("sl3 done")

    # Strategy Learner 4: Impact = 0.100
    learner4 = sl.StrategyLearner(verbose=False, impact=0.100)
    learner4.add_evidence(symbol=sym, sd=is_start_date, ed=is_end_date)
    df_trades4 = learner4.testPolicy(symbol=sym, sd=is_start_date, ed=is_end_date, sv=startval)
    port_val4 = mk.compute_portvals(df_trades4, start_val=startval, commission=commission, impact=0.100)
    is_cr4, is_avg4, is_sddr4, is_sr4 = mk.compute_stats(port_val4)
    trades_total4 = df_trades4[df_trades4 != 0].count()[0]
    #print("sl4 done")

    # Normalize Portfolio Values
    port_val1 = port_val1 / port_val1.iloc[0]
    port_val2 = port_val2 / port_val2.iloc[0]
    port_val3 = port_val3 / port_val3.iloc[0]
    port_val4 = port_val4 / port_val4.iloc[0]

    # Plot
    plt.figure(figsize=(14.5, 5))
    port_val1.plot(color="blue", label="Impact: 0.000")
    port_val2.plot(color="red", label="Impact: 0.025")
    port_val3.plot(color="green", label="Impact: 0.050")
    port_val4.plot(color="orange", label="Impact: 0.100")
    plt.title("Experiment 2: Strategy Learner With Varying Impact Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig('Experiment2.png')
    #plt.show()

    # Write to File
    file = open("p8_results_Experiment2.txt", "w")
    file.write("Dates: " + str(is_start_date) + " to " + str(is_end_date) + " for " + str(sym) +
               "\n" +
               "\nStrategy Learner - Impact 0.000" +
               "\nCumulative Return: " + str(is_cr1) +
               "\nStandard Deviation: " + str(is_sddr1) +
               "\nAverage Daily Return: " + str(is_avg1) +
               "\nSharpe Ratio: " + str(is_sr1) +
               "\nFinal Portfolio Value: " + str(port_val1[-1]) +
               "\nTotal Trades: " + str(trades_total1) +
               "\n" +
               "\nStrategy Learner - Impact 0.025" +
               "\nCumulative Return: " + str(is_cr2) +
               "\nStandard Deviation: " + str(is_sddr2) +
               "\nAverage Daily Return: " + str(is_avg2) +
               "\nSharpe Ratio: " + str(is_sr2) +
               "\nFinal Portfolio Value: " + str(port_val2[-1]) +
               "\nTotal Trades: " + str(trades_total2) +
               "\n" +
               "\nStrategy Learner - Impact 0.050" +
               "\nCumulative Return: " + str(is_cr3) +
               "\nStandard Deviation: " + str(is_sddr3) +
               "\nAverage Daily Return: " + str(is_avg3) +
               "\nSharpe Ratio: " + str(is_sr3) +
               "\nFinal Portfolio Value: " + str(port_val3[-1]) +
               "\nTotal Trades: " + str(trades_total3) +
               "\n" +
               "\nStrategy Learner - Impact 0.100" +
               "\nCumulative Return: " + str(is_cr4) +
               "\nStandard Deviation: " + str(is_sddr4) +
               "\nAverage Daily Return: " + str(is_avg4) +
               "\nSharpe Ratio: " + str(is_sr4) +
               "\nFinal Portfolio Value: " + str(port_val4[-1]) +
               "\nTotal Trades: " + str(trades_total4) +
               "\n" )
    file.close()


def author():
    return "mlukacsko3"


if __name__ == "__main__":
    np.random.seed(123456)
    main()
