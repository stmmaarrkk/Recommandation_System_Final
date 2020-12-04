import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
class LinearRegRecur:
    def __init__(self, evaluator, epoch=1, batchSize=float("inf")):
        self.evaluator = evaluator
        self.epoch = epoch
        self.batchSize = batchSize
        self.model = {}
        self.model["x"] = LinearRegression()
        self.model["y"] = LinearRegression()
        self.inputNodesPerSeq = None
        self.outputNodesPerSeq= None
    def prepareData(self, data, target):
        assert data.shape[2] == 2 and target.shape[2] == 2, "The third dim should be 2"
        self.inputNodesPerSeq, self.outputNodesPerSeq = data.shape[1], target.shape[1]

        xObsv, yObsv = data[:, :, 0].reshape(-1, self.inputNodesPerSeq), data[:, :, 1].reshape(-1,
                                                                                                 self.inputNodesPerSeq)
        xAns, yAns = target[:, :, 0].reshape(-1, self.outputNodesPerSeq), target[:, :, 1].reshape(-1,
                                                                                                    self.outputNodesPerSeq)
        # lastValue = []

        # for i in range(xObsv.shape[0]):
        #     lastValue.append((xObsv[i, -1], yObsv[i, -1]))
        #     xObsv[i, :] -= lastValue[-1][0]
        #     yObsv[i, :] -= lastValue[-1][1]
        #
        # for i in range(xAns.shape[0]):
        #     xAns[i, :] -= lastValue[i][0]
        #     yAns[i, :] -= lastValue[i][1]

        return xObsv, xAns, yObsv, yAns
    def fit(self, data, target):
        xObsv, xAns, yObsv, yAns = self.prepareData(data, target)
        N = xObsv.shape[0]
        for ep in range(self.epoch):
            print("-----epoch {}-----".format(ep))
            xObsvCascade = xObsv.copy()
            yObsvCascade = yObsv.copy()

            xAnsCascade = xAns[:, 0].copy().reshape(-1, 1) # is a ((rd+1) * nAns) by 1 array
            yAnsCascade = yAns[:, 0].copy().reshape(-1, 1)

            xNextPredict = None
            yNextPredict = None
            for rd in range(self.outputNodesPerSeq):
                if rd > 0:
                    # append predicted data
                    newXObsv = np.hstack((xObsvCascade[-N:, 1:], xNextPredict))
                    newYObsv = np.hstack((yObsvCascade[-N:, 1:], yNextPredict))
                    xObsvCascade = np.vstack((xObsvCascade, newXObsv))
                    yObsvCascade = np.vstack((yObsvCascade, newYObsv))
                    
                    xAnsCascade = np.vstack((xAnsCascade, xAns[:, rd].reshape(-1, 1)))
                    yAnsCascade = np.vstack((yAnsCascade, yAns[:, rd].reshape(-1, 1)))


                ##batch
                if self.batchSize * N < xObsvCascade.shape[0]:
                    #give up the first batchSize * N data
                    realSize = self.batchSize * N
                    xObsvCascade = xObsvCascade[-realSize:]
                    yObsvCascade = yObsvCascade[-realSize:]
                    xAnsCascade = xAnsCascade[-realSize:]
                    yAnsCascade = yAnsCascade[-realSize:]
                #assert xObsvCascade.shape[0] == (rd+1) * N, "# of samples should equal to (rd+1) * N)"


                #In the first round of k epoch(k >= 2), we should use the old model
                if not(ep >= 1 and rd == 0): #if not the first round and first epoch
                    self.model["x"].fit(xObsvCascade, xAnsCascade)
                    self.model["y"].fit(yObsvCascade, yAnsCascade)
                
                # #evaluate
                xNextPredict = self.model["x"].predict(xObsvCascade[-N:, :])
                yNextPredict = self.model["y"].predict(yObsvCascade[-N:, :])


            ##evaluate
            self.evaluate(xObsv, xAns, yObsv, yAns)

    def score(self, data, target):
        xObsv, xAns, yObsv, yAns = self.prepareData(data, target)
        self.evaluate(xObsv, xAns, yObsv, yAns)

    def evaluate(self, xObsv, xAns, yObsv, yAns):
        xPredict, yPredict = self.predict(xObsv.copy(), yObsv.copy())
        result = self.evaluator.RMSE(xPredict, yPredict, xAns, yAns)
        print("RMSE: {:.3f}\n".format(result))
    def predict(self, xObsv, yObsv):
        N = xObsv.shape[0]
        xPredict = np.zeros((N, self.outputNodesPerSeq), dtype=np.float)
        yPredict = np.zeros((N, self.outputNodesPerSeq), dtype=np.float)

        for i in range(self.outputNodesPerSeq):
            xPredict[:, i] = self.model["x"].predict(xObsv[:, i:(self.outputNodesPerSeq+i)]).reshape(-1, )
            yPredict[:, i] = self.model["y"].predict(yObsv[:, i:(self.outputNodesPerSeq+i)]).reshape(-1, )


            #append in the tail
            xObsv = np.hstack((xObsv, xPredict[:, i].reshape(-1, 1)))
            yObsv = np.hstack((yObsv, yPredict[:, i].reshape(-1, 1)))
        return xPredict, yPredict