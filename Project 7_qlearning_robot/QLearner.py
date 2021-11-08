""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import random as rand  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		   	 		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		   	 		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		   	 		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		   	 		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		   	 		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		   	 		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		   	 		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		   	 		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0  		  	   		   	 		  		  		    	 		 		   		 		  
        self.a = 0
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.tuple = []
        self.Q = np.zeros((num_states,num_actions))
        self.R = np.zeros((num_states,num_actions))
        self.T = np.zeros((num_states, num_actions, num_states))
        self.T_count = self.T.copy()
        self.T_count.fill(0.00001)



    def author(self):
        return "mlukacsko3"

  		  	   		   	 		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		   	 		  		  		    	 		 		   		 		  
        :type s: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        # action = rand.randint(0, self.num_actions - 1)
        # if self.verbose:
        #     print(f"s = {s}, a = {action}")
        self.s = s
        random = np.random.uniform(low=0, high=1)
        if self.rar > random:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])
        self.a = action
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		   	 		  		  		    	 		 		   		 		  

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        # action = rand.randint(0, self.num_actions - 1)
        # if self.verbose:
        #     print(f"s = {s_prime}, a = {action}, r={r}")
        # return action
        a = self.a
        s = self.s
        alpha = self.alpha
        gamma = self.gamma

        # Updating Q-Table
        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])])
        self.tuple.append((s, a, s_prime, r))

        # Implementing Dyna
        dyna_data = np.random.randint(len(self.tuple), size=self.dyna)

        for i in range(0, self.dyna):
            random_index = dyna_data[i]
            dyan_s, dyna_a, dyna_sprime, dyna_r = self.tuple[random_index]
            self.Q[dyan_s, dyna_a] = (1 - alpha) * self.Q[dyan_s, dyna_a] + alpha * (
                    dyna_r + gamma * self.Q[dyna_sprime, np.argmax(self.Q[dyna_sprime])])

        # Choose An Action - Random or Optimal
        random = np.random.uniform(low=0, high=1)
        if self.rar > random:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])

        # Update Action
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action
        return action


if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		   	 		  		  		    	 		 		   		 		  
