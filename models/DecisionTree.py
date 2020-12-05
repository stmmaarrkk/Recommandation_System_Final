import numpy as np
from sklearn.metrics import mean_squared_error
from helper import RMSE
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
import math

class DecisionTree:
    def __init__(self):
        self.model = {}
        self.model["x"] = MultiOutputRegressor(DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.13, random_state=4))
        self.model["y"] = MultiOutputRegressor(DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.13, random_state=4))
        self.inputNodesPerSeq = None
        self.outputNodesPerSeq= None

    
    def prepareData(self, data, target):
        assert data.shape[2] == 2 and target.shape[2] == 2, "The third dim should be 2"
        self.inputNodesPerSeq, self.outputNodesPerSeq = data.shape[1], target.shape[1]

        xTrain, yTrain = data[:, :, 0].reshape(-1, self.inputNodesPerSeq), data[:, :, 1].reshape(-1,
                                                                                                 self.inputNodesPerSeq)
        xTest, yTest = target[:, :, 0].reshape(-1, self.outputNodesPerSeq), target[:, :, 1].reshape(-1,
                                                                                                    self.outputNodesPerSeq)
        return xTrain, xTest, yTrain, yTest
    
    def fit(self, data, target):
        xTrain, xTest, yTrain, yTest = self.prepareData(data, target)

        self.model["x"].fit(xTrain, xTest)
        self.model["y"].fit(yTrain, yTest)

        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)
        result = RMSE(xPredict, yPredict, xTest, yTest)
        print("RMSE(training set): {:.3f}".format(result))
    
    def score(self, data, target):
        xTrain, xTest, yTrain, yTest = self.prepareData(data, target)

        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)
        result = RMSE(xPredict, yPredict, xTest, yTest)
        print("RMSE(testing set): {:.3f}".format(result))
        #mean_squared_error(y_true, y_pred, multioutput='raw_values')