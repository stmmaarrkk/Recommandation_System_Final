import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
import math

class SVM:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.model = {}
        tuned_parameters = [{'kernel':('linear', 'rbf'), 'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                     'C': [1, 10, 100, 1000]}]
        GS = GridSearchCV( SVR(), tuned_parameters, scoring = 'neg_root_mean_squared_error')
        # self.model["x"] = MultiOutputRegressor(SVR(kernel='rbf', C= 1e2, epsilon=0.1))
        # self.model["y"] = MultiOutputRegressor(SVR(kernel='rbf', C= 1e2, epsilon=0.1))
        self.model["x"] = MultiOutputRegressor(GS)
        self.model["y"] = MultiOutputRegressor(GS)
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
        result = self.evaluator.RMSE(xPredict, yPredict, xTest, yTest)
        print("RMSE(training set): {:.3f}".format(result))
    
    def score(self, data, target):
        xTrain, xTest, yTrain, yTest = self.prepareData(data, target)

        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)
        result = self.evaluator.RMSE(xPredict, yPredict, xTest, yTest)
        print("RMSE(testing set): {:.3f}".format(result))
        #mean_squared_error(y_true, y_pred, multioutput='raw_values')