import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegNorm:
    # normalize with max and min of each row
    def __init__(self):
        self.model = {}
        self.model["x"] = LinearRegression()
        self.model["y"] = LinearRegression()
        self.inputNodesPerSeq = None
        self.outputNodesPerSeq = None
    def prepareData(self, data, target):
        self.inputNodesPerSeq, self.outputNodesPerSeq = data.shape[1], target.shape[1]
        xTrain, yTrain = data[:, :, 0].copy().reshape(-1, self.inputNodesPerSeq), data[:, :, 1].copy().reshape(-1,
                                                                                                 self.inputNodesPerSeq)
        xTest, yTest = target[:, :, 0].copy().reshape(-1, self.outputNodesPerSeq), target[:, :, 1].copy().reshape(-1,
                                                                                                    self.outputNodesPerSeq)
        # x, xDenomFactor = normalization(np.hstack((xTrain, xTest)))
        # y, yDenomFactor = normalization(np.hstack((yTrain, yTest)))
        # yTrain, yTestNorm = y[:, :self.inputNodesPerSeq], y[:, self.inputNodesPerSeq:]
        # xTrain, xTestNorm = x[:, :self.inputNodesPerSeq], x[:, self.inputNodesPerSeq:]
        xTrain, xDenomFactor = normalization(xTrain)
        yTrain, yDenomFactor = normalization(yTrain)
        xTestNorm = normalizationWithDenomFactor(xTest.copy(), xDenomFactor)
        yTestNorm = normalizationWithDenomFactor(yTest.copy(), yDenomFactor)
        return xTrain, xTest, xTestNorm, yTrain, yTest, yTestNorm, xDenomFactor, yDenomFactor
    def fit(self, data, target):
        xTrain, xTest, xTestNorm, yTrain, yTest, yTestNorm, xDenomFactor, yDenomFactor = self.prepareData(data, target)

        self.model["x"].fit(xTrain, xTestNorm) #use norm data to fit
        self.model["y"].fit(yTrain, yTestNorm) #use norm data to fit
        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)

        ##denormalization##
        xPredict = denormalization(xPredict, xDenomFactor)
        yPredict = denormalization(yPredict, yDenomFactor)

        result = RMSE(xPredict, yPredict, xTest, yTest) #use orginal data to validate
        print("RMSE(training set): {:.3f}".format(result))
    def predict(self, data):
        pass
    def score(self, data, target):
        xTrain, xTest, xTestNorm, yTrain, yTest, yTestNorm, xDenomFactor, yDenomFactor = self.prepareData(data, target)
        xPredict = self.model["x"].predict(xTrain)
        yPredict = self.model["y"].predict(yTrain)


        xPredict = denormalization(xPredict, xDenomFactor)
        yPredict = denormalization(yPredict, yDenomFactor)

        result = RMSE(xPredict, yPredict, xTest, yTest)
        print("RMSE(testing set): {:.3f}".format(result))

def normalization(data):
    denomFactor = []
    for i in range(data.shape[0]):
        denomFactor.append([min(data[i]), max(data[i]) - min(data[i])])
        if denomFactor[-1][1] == 0: #if max == min, make denominator to 1
            denomFactor[-1][1] = 1

        data[i, :] = (data[i, :] - denomFactor[-1][0]) / denomFactor[-1][1]
    return data, np.array(denomFactor)

def normalizationWithDenomFactor(data, denomFactor):
    assert data.shape[0] == denomFactor.shape[0]
    for i in range(data.shape[0]):
        assert denomFactor[i, 1] != 0, "value cannot divided by 0"

        data[i, :] = (data[i, :] - denomFactor[i, 0]) / denomFactor[i, 1]
    return data

def denormalization(data, denomFactor):
    assert data.shape[0] == denomFactor.shape[0]
    for i in range(data.shape[0]):
        assert denomFactor[i, 1] != 0, "value cannot divided by 0"
        data[i, :] = data[i, :] * denomFactor[i, 1] + denomFactor[i, 0]
    return data