import numpy as np
import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose = False):
        self.verbose = verbose
        self.learner = learner
        self.bags = bags
        self.kwargs = kwargs
        self.learners = [learner(**kwargs) for i in range(0, bags)]
        pass

    def addEvidence(self, data_x, data_y):
        data = data_x.shape[0]
        for i in self.learners:
            index = np.random.choice(data, data)
            index_bag_x = data_x[index]
            index_bag_y = data_y[index]
            i.addEvidence(index_bag_x, index_bag_y)

    def query(self, points):
        predict = np.array([learners.query(points) for learners in self.learners])
        return np.mean(predict, axis=0)

    def author(self):
        return 'mlukacsko3'
