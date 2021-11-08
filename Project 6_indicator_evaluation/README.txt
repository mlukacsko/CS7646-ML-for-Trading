README
---------------------------
Project Overview:
The files included in the indicator_evaluation folder serves as an instrument to develop an understanding of the maximum amount that can be earned through trading given a specific stock.  This is done analyzing the results of a benchmark strategy as well as an optimal strategy.  Moreover, 5 trading indicators are developed and examine how they might be used to generate trading signals.  All stock data used is historical JPMorgan Chase & Co (NYSE: JPM) stock data.
---------------------------
Contents and Description:
1] TheoreticallyOptimalStrategy.py – Python file that return a Pandas DataFrame, df_trades, containing an index of trading days and a ‘Holding’ column with stock sale information.  A positive value in the ‘Holding’ column signals a buy for JPM, a negative value sells JPM and 0 holds.
2] marketsimcode.py – Python file that accepts a Pandas DataFrame, orders, and returns the corresponding portfolio value is a df_prt_val DataFrame.  Additionally, this file calculates and returns the daily return, cumulative daily return, average daily return, and standard deviation daily return in a text file.
3] indicators.py – Python file that calculates various trading indicators that could signal advantageous opportunities to buy/sell a stock.  Each indicator is subsequently plotted and saved as a .png file.  5 indicators are contained in this file
4] testproject.py – Python file that acts as the “main” method.  This file is the entry point to the project, making calls to all other Python files as needed.  When ran, this file outputs the benchmark/optimal strategies daily return values to a text file and plots the normalized portfolio values.  Because of the call to indicators.py, you can expect the .png files generated in indicators.py to be present after successfully running this testproject.py file.
---------------------------
How to Run the Code:
1] To run this code from a terminal, navigate to the indicator_evaluation folders location
2] From the /home/[yourUser]/…./indicator_evaluation directory, type in PYTHONPATH=../:. python testproject.py
3] Once the file is complete, you will see the required .png image files containing the required graphs, as well as a text file containing the statistic for each trading strategy described in the overview.
4] Each individual file can be ran on its own as well, if necessary.  From the /home/[yourUser]/…./indicator_evaluation folder, enter “PYTHONPATH=../:. python [yourFileName].py.
