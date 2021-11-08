""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
import warnings
import numpy as np
from operator import itemgetter
  		  	   		   	 		  		  		    	 		 		   		 		  

class DTLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		   	 		  		  		    	 		 		   		 		  
    your own correct DTLearner from Project 3.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if len(np.unique(data_y)) == 1 or data_x.shape[0] <= self.leaf_size:
            return np.array([-1, data_y.mean(), np.nan, np.nan])
        else:
            bf = self.get_best_feature(data_x, data_y)
            sv = self.get_split_value(data_x, bf)
            start_left = data_x[:, bf] <= sv
            if np.all(start_left):
                return np.array([-1, data_y.mean(), np.nan, np.nan])
            left_tree = self.build_tree(data_x[start_left], data_y[start_left])
            right_tree = self.build_tree(data_x[start_left != True], data_y[start_left != True])
            if left_tree.ndim <= 1:
                right_tree_start = 2
            if left_tree.ndim > 1:
                right_tree_start = left_tree.shape[0] + 1
            root = np.array([bf, sv, 1, right_tree_start])
            tree = np.vstack((root, left_tree, right_tree))
            return tree

    def query(self, points):
        train_y = []
        for n in points:
            train_y.append(self.dtree_search(n, row=0))
        return np.asarray(train_y)

    def get_best_feature(self, data_x, data_y):
        temp = []
        for r in range(data_x.shape[1]):
            corr = np.corrcoef(data_x[:, r], data_y)
            absolute = abs(corr[0, 1])
            temp.append((r, absolute))
        best_feature = max(temp, key=itemgetter(1))[0]
        return best_feature

    def get_split_value(self, data_x, bf):
        split_value = np.median(data_x[:, bf])
        return split_value

    def dtree_search(self, num, row):
        feature, split_value = self.tree[row, 0:2]
        if feature == -1:
            return split_value
        elif num[int(feature)] <= split_value:
            predicted_value = self.dtree_search(num, row + int(self.tree[row, 2]))
        else:
            predicted_value = self.dtree_search(num, row + int(self.tree[row, 3]))
        return predicted_value

    def author(self):
        return "mlukacsko3"

if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		   	 		  		  		    	 		 		   		 		  
