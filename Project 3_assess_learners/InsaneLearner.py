import BagLearner as bl
import LinRegLearner as lr
import numpy as np


class InsaneLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = [bl.BagLearner(learner=lr.LinRegLearner, bags=20, kwargs={}, boost = False) for i in range(20)]

    def addEvidence(self, data_x, data_y):
        for learners in self.learners:
            learners.addEvidence(data_x, data_y)

    def query(self, points):
        predict = np.array([learners.query(points) for learners in self.learners])
        return np.mean(predict, axis=0)

    def author(self):
        return 'mlukacsko3'