import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
class LinearReg:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.model = {}
        self.model["x"] = LinearRegression()
        self.model["y"] = LinearRegression()
        self.inputNodesPerSeq = None
        self.outputNodesPerSeq= None
    def prepareData(self, data, target):
        assert data.shape[2] == 2 and target.shape[2] == 2, "The third dim should be 2"
        self.inputNodesPerSeq, self.outputNodesPerSeq = data.shape[1], target.shape[1]

        xTrain, yTrain = data[:, :, 0].copy().reshape(-1, self.inputNodesPerSeq), data[:, :, 1].copy().reshape(-1,
                                                                                                 self.inputNodesPerSeq)
        xTest, yTest = target[:, :, 0].copy().reshape(-1, self.outputNodesPerSeq), target[:, :, 1].copy().reshape(-1,
                                                                                                    self.outputNodesPerSeq)
        # lastValue = []

        # for i in range(xTrain.shape[0]):
        #     lastValue.append((xTrain[i, -1], yTrain[i, -1]))
        #     xTrain[i, :] -= lastValue[-1][0]
        #     yTrain[i, :] -= lastValue[-1][1]
        #
        # for i in range(xTest.shape[0]):
        #     xTest[i, :] -= lastValue[i][0]
        #     yTest[i, :] -= lastValue[i][1]

        return xTrain, xTest, yTrain, yTest
    def fit(self, data, target):
        xTrain, xTest, yTrain, yTest = self.prepareData(data, target)

        self.model["x"].fit(xTrain, xTest)
        self.model["y"].fit(yTrain, yTest)
        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)
        rmse = self.evaluator.RMSE(xPredict, yPredict, xTest, yTest)
        fmse = self.evaluator.FMSE(xPredict, yPredict, xTest, yTest)
        print("RMSE(training set): {:.3f}".format(rmse))
        print("FMSE(training set): {:.3f}".format(fmse))
        #mean_squared_error(y_true, y_pred, multioutput='raw_values')
    def score(self, data, target):
        xTrain, xTest, yTrain, yTest = self.prepareData(data, target)

        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)
        rmse = self.evaluator.RMSE(xPredict, yPredict, xTest, yTest)
        fmse = self.evaluator.FMSE(xPredict, yPredict, xTest, yTest)
        print("RMSE(testing set): {:.3f}".format(rmse))
        print("FMSE(testing set): {:.3f}".format(fmse))

