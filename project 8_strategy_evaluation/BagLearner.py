import RTLearner as rt
import numpy as np
from scipy import stats

class BagLearner(object):

    def __init__(self, learner = rt.RTLearner, kwargs = {'leaf_size': 1}, bags = 20, boost = False, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.verbose = verbose
        self.learners = [learner(**kwargs) for i in range(0, bags)]

    def add_evidence(self,  data_x, data_y):
        data = data_x.shape[0]
        for i in range(self.bags):
            index = np.random.choice(data, data)
            index_bag_x = data_x[index]
            index_bag_y = data_y[index]
            self.learners[i].addEvidence(index_bag_x, index_bag_y)

    def query(self, points):
        predict_y = np.ones([self.bags, points.shape[0]])
        for j in range(self.bags):
            predict_y[j] = self.learners[j].query(points)
        predict_y_modal = stats.mode(predict_y)[0]
        return predict_y_modal

    def author(self):
        return 'mlukacsko3'
