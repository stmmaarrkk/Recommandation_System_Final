import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from helper import RMSE
class LinearReg:
    def __init__(self):
        self.model = {}
        self.model["x"] = LinearRegression()
        self.model["y"] = LinearRegression()
        self.inputNodesPerSeq = None
        self.outputNodesPerSeq= None
    def fit(self, data, target):
        assert data.shape[2] == 2 and target.shape[2] == 2 , "The third dim should be 2"
        # assert data.shape[1] == self.inputNodesPerSeq, "The third dim should be equal to inputNodesPerSeq"
        # assert target.shape[1] == self.outputNodesPerSeq, "The third dim should be equal to outputNodesPerSeq"
        self.inputNodesPerSeq, self.outputNodesPerSeq = data.shape[1], target.shape[1]

        xTrain, yTrain = data[:, :, 0].reshape(-1, self.inputNodesPerSeq), data[:, :, 1].reshape(-1, self.inputNodesPerSeq)
        xTest, yTest = target[:, :, 0].reshape(-1, self.outputNodesPerSeq), target[:, :, 1].reshape(-1, self.outputNodesPerSeq)
        self.model["x"].fit(xTrain, xTest)
        self.model["y"].fit(yTrain, yTest)
        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)
        result = RMSE(xPredict, yPredict, xTest, yTest)
        print(result)
        #mean_squared_error(y_true, y_pred, multioutput='raw_values')
