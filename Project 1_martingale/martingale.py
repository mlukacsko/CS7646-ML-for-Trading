""""""
"""Assess a betting strategy.  		  	   		   	 		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  

Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  

We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  

-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  

Student Name: Michael Lukacsko  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mlukacsko3"  # replace tb34 with your Georgia Tech username.


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903714876  # replace with your GT ID number


def get_spin_result(win_prob):
    """
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.

    :param win_prob: The probability of winning
    :type win_prob: float
    :return: The result of the spin.
    :rtype: bool
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def gambling_simulator(win_prob, unlimited_bankroll):
    winnings_array = np.full((1000),80)
    episode_winnings = 0
    counter = 0

    while episode_winnings < 80:
        won = False
        bet_amount = 1

        while not won:
            if counter >= 1000:
                return winnings_array

            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings += bet_amount

            elif won == False:
                episode_winnings -= bet_amount
                bet_amount *= 2
            winnings_array[counter] = episode_winnings
            counter += 1

            if unlimited_bankroll == False:
                bankroll = 256
                if episode_winnings == -bankroll:
                    winnings_array[counter:] = episode_winnings
                    return winnings_array
                if episode_winnings - bet_amount < -bankroll:
                    bet_amount = bankroll + episode_winnings


    return winnings_array


def experiment1_figure1(win_prob, unlimited_bankroll):
    for i in range(10):
        winnings = gambling_simulator(win_prob, unlimited_bankroll)
        plt.plot(winnings, label='Episode ' + str(i+1))
    plt.axis((0, 300, -256, 100))
    plt.title("Figure 1: 10 Episodes, Track of Winnings with Unlimited Bankroll")
    plt.xlabel('# of Spins ')
    plt.ylabel('$ of Winnings')
    plt.legend(loc='lower right')
    plt.savefig('Experiment1_Figure1.png')
    plt.clf()


def experiment1_figure2(win_prob, unlimited_bankroll):
    fig2_array = np.zeros((1000,1000))
    for j in range(1000):
        current = gambling_simulator(win_prob, unlimited_bankroll)
        fig2_array[j] = current

    mean_of_winnings = np.mean(fig2_array, axis = 0)
    std_of_winnings = np.std(fig2_array, axis=0)
    mean_plus_std = mean_of_winnings + std_of_winnings
    mean_minus_std = mean_of_winnings - std_of_winnings
    plt.axis((0, 300, -256, 100))
    plt.title("Figure 2: 1000 Episodes, Mean of Winnings with Unlimited Bankroll")
    plt.xlabel('# of Spins ')
    plt.ylabel('$ of Winnings')
    plt.plot(mean_of_winnings, label="Mean of Winnings")
    plt.plot(mean_plus_std, label="Mean + std")
    plt.plot(mean_minus_std, label="Mean - std")
    plt.legend(loc='lower right')
    plt.savefig('Experiment1_Figure2.png')
    plt.clf()


def experiment1_figure3(win_prob, unlimited_bankroll):
    fig3_array = np.zeros((1000, 1000))
    for j in range(1000):
        current = gambling_simulator(win_prob, unlimited_bankroll)
        fig3_array[j] = current

    median_of_winnings = np.median(fig3_array, axis=0)
    std_of_winnings = np.std(fig3_array, axis=0)
    median_plus_std = median_of_winnings + std_of_winnings
    median_minus_std = median_of_winnings - std_of_winnings
    plt.axis((0, 300, -256, 100))
    plt.title("Figure 3: 1000 Episodes, Median of Winnings with Unlimited Bankroll)")
    plt.xlabel('# of Spins ')
    plt.ylabel('$ of Winnings')
    plt.plot(median_of_winnings, label="Median of Winnings")
    plt.plot(median_plus_std, label="Median + std")
    plt.plot(median_minus_std, label="Median - std")
    plt.legend(loc='lower right')
    plt.savefig('Experiment1_Figure3.png')
    plt.clf()


def experiment2_figure4(win_prob, unlimited_bankroll):
    fig4_array = np.zeros((1000,1000))
    for i in range(1000):
        current = gambling_simulator(win_prob, unlimited_bankroll)
        fig4_array[i] = current
    mean_of_winnings = np.mean(fig4_array, axis=0)
    std_of_winnings = np.std(fig4_array, axis=0)
    mean_plus_std = mean_of_winnings + std_of_winnings
    mean_minus_std = mean_of_winnings - std_of_winnings
    plt.axis((0, 300, -256, 100))
    plt.title("Figure 4: 1000 Episodes, Mean of Winnings with $256 Bankroll")
    plt.xlabel('# of Spins ')
    plt.ylabel('$ of Winnings')
    plt.plot(mean_of_winnings, label="Mean of Winnings")
    plt.plot(mean_plus_std, label="Mean + std")
    plt.plot(mean_minus_std, label="Mean - std")
    plt.legend(loc='lower left')
    plt.savefig('Experiment2_Figure4.png')
    plt.clf()


def experiment2_figure5(win_prob, unlimited_bankroll):
    fig5_array = np.zeros((1000, 1000))
    for j in range(1000):
        current = gambling_simulator(win_prob, unlimited_bankroll)
        fig5_array[j] = current

    median_of_winnings = np.median(fig5_array, axis=0)
    std_of_winnings = np.std(fig5_array, axis=0)
    median_plus_std = median_of_winnings + std_of_winnings
    median_minus_std = median_of_winnings - std_of_winnings
    plt.axis((0, 300, -256, 100))
    plt.title("Figure 5: 1000 Episodes, Median of Winnings with $256 Bankroll)")
    plt.xlabel('# of Spins ')
    plt.ylabel('$ of Winnings')
    plt.plot(median_of_winnings, label="Median of Winnings")
    plt.plot(median_plus_std, label="Median + std")
    plt.plot(median_minus_std, label="Median - std")
    plt.legend(loc='lower left')
    plt.savefig('Experiment2_Figure5.png')
    plt.clf()


def test_code():
    """
    Method to test your code
    """
    win_prob = (18 / 38)  # Probability of winning on black
    np.random.seed(gtid())  # do this only once
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    experiment1_figure1(win_prob, unlimited_bankroll=True)
    experiment1_figure2(win_prob, unlimited_bankroll=True)
    experiment1_figure3(win_prob, unlimited_bankroll=True)
    experiment2_figure4(win_prob, unlimited_bankroll=False)
    experiment2_figure5(win_prob, unlimited_bankroll=False)


if __name__ == "__main__":
    test_code()

