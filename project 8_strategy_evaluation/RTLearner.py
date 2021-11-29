import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def addEvidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if data_y.shape[0] <= self.leaf_size:
            return (np.array([[-1, np.mean(data_y), np.nan, np.nan]]))

        bf = self.get_best_feature(data_x, data_y)
        sv = self.get_split_value(data_x, bf)
        if (data_x[data_x[:, bf] <= sv]).shape[0] == data_x.shape[0]:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        if (data_x[data_x[:, bf] > sv]).shape[0] == data_x.shape[0]:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        left_tree = self.build_tree(data_x[data_x[:, bf] <= sv], data_y[data_x[:, bf] <= sv])
        right_tree = self.build_tree(data_x[data_x[:, bf] > sv], data_y[data_x[:, bf] > sv])

        root = np.array([[bf, sv, 1, left_tree.shape[0] + 1]])
        tree = (np.vstack((root, left_tree, right_tree)))
        return tree

    def query(self, points):
        predict_y = np.zeros(shape=(points.shape[0],))
        row = 0
        for data in points:
            i = 0
            array = self.tree[i]
            while array[0] != -1:
                column_index = int(array[0])
                if data[column_index] <= array[1]:
                    i += int(array[2])
                    array = self.tree[i]
                else:
                    i += int(array[3])
                    array = self.tree[i]
            predict_y[row] = (array[1])
            row = row + 1
        return predict_y

    def get_best_feature(self, data_x, data_y):
        bf = np.random.randint(0, data_x.shape[1])
        return bf

    def get_split_value(self, data_x, bf):
        sv = np.median(data_x[:, bf])
        return sv

    def author(self):
        return 'mlukacsko3'
