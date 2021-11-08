
import math
import matplotlib.pyplot as plt
import numpy as np
import LinRegLearner as lrl
import DTLearner as dtl
import BagLearner as bl
import RTLearner as rtl
import util

if __name__ == "__main__":
    # if len(sys.argv) != 2:
        # print("Usage: python testlearner.py <filename>")
        # sys.exit(1)
    data = np.genfromtxt(util.get_learner_data_file('Istanbul.csv'), delimiter=',')
    data = data[1:, 1:]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")

    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it

    '''Experiment 1'''

    in_sample = np.zeros((75, 1))
    out_samples = np.zeros((75, 1))
    for x in range(1, 76):
        dt_learner = dtl.DTLearner(leaf_size=x, verbose=False)  # create a DTdt_Learner
        dt_learner.addEvidence(train_x, train_y)  # train it
        # print(dt_learner.author())

        # evaluate in sample
        prediction = dt_learner.query(train_x)  # get the predictions
        dt_rmse = math.sqrt(((train_y - prediction) ** 2).sum() / train_y.shape[0])
        in_sample[x - 1, 0] = dt_rmse
        c = np.corrcoef(prediction, y=train_y)
        # print()
        # print("In sample results")
        # print(f"dt_RMSE: {dt_rmse}")
        c = np.corrcoef(prediction, y=train_y)
        # print(f"corr: {c[0,1]}")

        # evaluate out of sample
        prediction = dt_learner.query(test_x)  # get the predictions
        dt_rmse = math.sqrt(((test_y - prediction) ** 2).sum() / test_y.shape[0])
        out_samples[x - 1, 0] = dt_rmse
        c = np.corrcoef(prediction, y=test_y)
        # print()
        # print("Out of sample results")
        # print(f"dt_RMSE: {dt_rmse}")
        c = np.corrcoef(prediction, y=test_y)
        # print(f"corr: {c[0,1]}")
    # plot data
    x_axis = np.arange(1, 76)
    plt.plot(x_axis, in_sample, label="In Sample dt_RMSE")
    plt.plot(x_axis, out_samples, label="Out of Samples dt_RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("dt_RMSE")
    plt.legend()
    plt.title("Exp 1: DTdt_Learner dt_RMSE Analysis")
    plt.grid(True)
    plt.savefig("Experiment 1.png")
    plt.clf()

    '''Experiment 2'''

    rmseBagArr1 = np.zeros((75, 1))
    rmseBagArr2 = np.zeros((75, 1))
    rmseRTArr1 = np.zeros((75, 1))
    rmseRTArr2 = np.zeros((75, 1))
    for x in range(1, 76):
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": x}, bags=10, boost=False,
                                verbose=False)
        learnerRT = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": x}, bags=10, boost=False,
                                  verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        learnerRT.add_evidence(train_x, train_y)
        predYBag = learner.query(train_x)  # get the predictions
        predYRT = learnerRT.query(train_x)
        rmseBag1 = math.sqrt(((train_y - predYBag) ** 2).sum() / train_y.shape[0])
        rmseRT1 = math.sqrt(((train_y - predYRT) ** 2).sum() / train_y.shape[0])
        rmseBagArr1[x - 1, 0] = rmseBag1
        rmseRTArr1[x - 1, 0] = rmseRT1
        # evaluate out of sample
        predYBag = learner.query(test_x)  # get the predictions
        predYDT = learnerRT.query(test_x)
        rmseBag2 = math.sqrt(((test_y - predYBag) ** 2).sum() / test_y.shape[0])
        rmseRT2 = math.sqrt(((test_y - predYDT) ** 2).sum() / test_y.shape[0])
        rmseBagArr2[x - 1, 0] = rmseBag2
        rmseRTArr2[x - 1, 0] = rmseRT2

    xaxis = np.arange(1, 76)
    line1, = plt.plot(xaxis, rmseBagArr1, label="In sample(DTBag)")
    line2, = plt.plot(xaxis, rmseBagArr2, label="Out of Sample(DTBag)")
    # line3, = plt.plot(xaxis, rmseRTArr1, label="In sample(RTBag)")
    # line4, = plt.plot(xaxis, rmseRTArr2, label="Out of Sample(RTBag)")
    plt.legend()
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.title("Exp 2: DT BagLearner RMSE Analysis")
    plt.grid(True)
    # plt.show()
    plt.savefig("Experiment 2.png")
    plt.clf()

    '''Experiment 3'''

    rmseBagArr1 = np.zeros((75, 1))
    rmseBagArr2 = np.zeros((75, 1))
    rmseRTArr1 = np.zeros((75, 1))
    rmseRTArr2 = np.zeros((75, 1))
    for x in range(1, 76):
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": x}, bags=10, boost=False,
                                verbose=False)
        learnerRT = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": x}, bags=10, boost=False,
                                  verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        learnerRT.add_evidence(train_x, train_y)
        predYBag = learner.query(train_x)  # get the predictions
        predYRT = learnerRT.query(train_x)
        rmseBag1 = math.sqrt(((train_y - predYBag) ** 2).sum() / train_y.shape[0])
        rmseRT1 = math.sqrt(((train_y - predYRT) ** 2).sum() / train_y.shape[0])
        rmseBagArr1[x - 1, 0] = rmseBag1
        rmseRTArr1[x - 1, 0] = rmseRT1
        # evaluate out of sample
        predYBag = learner.query(test_x)  # get the predictions
        predYDT = learnerRT.query(test_x)
        rmseBag2 = math.sqrt(((test_y - predYBag) ** 2).sum() / test_y.shape[0])
        rmseRT2 = math.sqrt(((test_y - predYDT) ** 2).sum() / test_y.shape[0])
        rmseBagArr2[x - 1, 0] = rmseBag2
        rmseRTArr2[x - 1, 0] = rmseRT2

    xaxis = np.arange(1, 76)
    line1, = plt.plot(xaxis, rmseBagArr1, label="In sample(DTBag)")
    line2, = plt.plot(xaxis, rmseBagArr2, label="Out of Sample(DTBag)")
    line3, = plt.plot(xaxis, rmseRTArr1, label="In sample(RTBag)")
    line4, = plt.plot(xaxis, rmseRTArr2, label="Out of Sample(RTBag)")
    plt.legend()
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.title("Exp 3: BagLearner RMSE Analysis")
    plt.grid(True)
    # plt.show()
    plt.savefig("Experiment 3.png")
    plt.clf()

    """Experiment 4"""

    inSamples = np.zeros((75, 1))
    outSamples = np.zeros((75, 1))
    inSamplesRT = np.zeros((75, 1))
    outSamplesRT = np.zeros((75, 1))
    for x in range(1, 76):
        learner = dtl.DTLearner(leaf_size=x, verbose=False)  # create a DTLearner
        learnerRT = rtl.RTLearner(leaf_size=x, verbose=False)
        learner.add_evidence(train_x, train_y)  # train it
        learnerRT.add_evidence(train_x, train_y)
        # print(learner.author())

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        pred_yRT = learnerRT.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmseRT = math.sqrt(((train_y - pred_yRT) ** 2).sum() / train_y.shape[0])
        inSamples[x - 1, 0] = rmse
        inSamplesRT[x - 1, 0] = rmseRT
        c = np.corrcoef(pred_y, y=train_y)
        # print()
        # print("In sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=train_y)
        # print(f"corr: {c[0,1]}")

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        pred_yRT = learnerRT.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        rmseRT = math.sqrt(((test_y - pred_yRT) ** 2).sum() / test_y.shape[0])
        outSamples[x - 1, 0] = rmse
        outSamplesRT[x - 1, 0] = rmseRT
        c = np.corrcoef(pred_y, y=test_y)
        # print()
        # print("Out of sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=test_y)
        # print(f"corr: {c[0,1]}")

    # plot data
    xaxis = np.arange(1, 76)
    line1, = plt.plot(xaxis, inSamples, label="In sample(DT)")
    line2, = plt.plot(xaxis, outSamples, label="Out of Sample(DT)")
    line3, = plt.plot(xaxis, inSamplesRT, label="In sample(RT)")
    line4, = plt.plot(xaxis, outSamplesRT, label="Out of Sample(RT)")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Exp 4: Single Learner RMSE Analysis")
    plt.grid(True)
    # plt.show()
    plt.savefig("Experiment 4.png")
    plt.clf()
