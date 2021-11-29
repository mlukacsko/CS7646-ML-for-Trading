README
----------------------------
Project Overview:
The files included in the strategy_evaluation folder server as a program to compare trading strategies and evaluate the results.
----------------------------
Contents and Description:
1] ManualStrategy.py – Python file that uses in sample and out of sample historical data to compare a manual trading strategy against a benchmark trading strategy. This file generates (1) a MS-InSample.png comparing the benchmark to manual learner using in sample data, (2) a MS-OutOfSample.png comparing the benchmark to manual learner using out of sample data, (3) Indicators.png which plots all the indicator values, and (4) p8_results_ManualStrategy.txt containing the learner statistics
2] StrategyLearner.py – Python file that implements a strategy learner utilizing RTLearner.py and BagLearner.py. 
3] indicator.py – Python file called by both ManualStrategy.py and StrategyLearner.py. This file returns the trading indicator values used to develop trading rules.
4] marketsimcode.py – Python file called by both ManualStrategy.py and StrategyLearner.py that calculates the portfolio value and computes the portfolio statistics. 
5] BagLearner.py – Python file used by StrategyLearner.py to implement the ensemble learner/bag learner.
6] RTLearner.py – Python file called by BagLearner.py to implement a learner using random trees
7] experiment1.py – Python file that compares the performance of a manual trading strategy, a benchmark trading strategy, and a strategy learner. This file generates (1) Experiment1.png plotting the manual strategy, benchmark strategy, and strategy learner performance and (2) p8_results_Experiment1.txt containing the different trading strategy portfolio statistics.
8] experiment2.py – Python file that compares the strategy learner using different values of impact. This file generates (1) Experiment2.png plotting each strategy learner’s performance using different impact values, and (2) p8_results_Experiment2.txt containing each strategy portfolio statistics.
9] testproject.py – Python file that drives the program. This file calls the main() method of ManualStrategy.py, experiment1.py, and experiment2.py.  
----------------------------
How to Run the Code:
1] To run this code from a terminal, navigate to the strategy_evaluation folders location
2] From the /home/[yourUser]/…./strategy_evaluation directory, type in PYTHONPATH=../:. python testproject.py
3] Once the program execution is complete, you will see the required .png image files containing the required graphs, as well as text files containing the statistic for each trading strategy described in the contents and description section above.
Note] No Additional parameters are note needed to call testproject.py.
