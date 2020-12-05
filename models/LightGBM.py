import numpy as np
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import math

class LightGBM:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.model = {}
        hyper_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': ['l2', 'auc'],
            'learning_rate': 0.005,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.7,
            'bagging_freq': 10,
            'verbose': 0,
            "max_depth": 8,
            "num_leaves": 12,  
            "max_bin": 512,
            "n_estimators": 1000
        }
        params = {
            'boosting_type' : 'gbdt',
            'objective' :'regression',
            'max_depth' : 4,
            'num_leaves' : 15,
            'learning_rate' : 0.1,
            'feature_fraction' : 0.9,
            'bagging_fraction' : 0.8,
            'bagging_freq' : 5,
            'min_data_in_leaf' : 12,
            'lambda_l2' : 0.1,
            'verbose' : 0
        }

        rf_params = {
            'boosting_type' : 'dart',
            'objective' :'regression',
            'max_depth' : 12,
            'num_leaves' : 1000,
            'learning_rate' : 0.1,
            'feature_fraction' : 0.9,
            'bagging_fraction' : 0.8,
            'bagging_freq' : 5,
            'min_data_in_leaf' : 10,
            'lambda_l2' : 0.1,
            'verbose' : 0
        }

        self.model["x"] = MultiOutputRegressor(lgb.LGBMRegressor(**params))
        self.model["y"] = MultiOutputRegressor(lgb.LGBMRegressor(**params))
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