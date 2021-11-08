import numpy as np
import random


class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def addEvidence(self, data_x, data_y):
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
            train_y.append(self.rtree_search(n, row=0))
        return np.asarray(train_y)

    def get_best_feature(self, data_x, data_y):
        correlation = []
        for r in range(data_x.shape[1]):
            corr = np.corrcoef(data_x[:, r], data_y)
            abscorr = abs(corr[0, 1])
            correlation.append((r, abscorr))
        best_feature = random.randint(0, data_x.shape[1] - 1)
        return best_feature

    def get_split_value(self, data_x, bf):
        split_value = np.median(data_x[:, bf])
        return split_value

    def rtree_search(self, num, row):
        feature, split_value = self.tree[row, 0:2]
        if feature == -1:
            return split_value
        elif num[int(feature)] <= split_value:  # If less, go left
            predicted_value = self.rtree_search(num, row + int(self.tree[row, 2]))
        else:
            predicted_value = self.rtree_search(num, row + int(self.tree[row, 3]))
        return predicted_value

    def author(self):
        return "mlukacsko3"






