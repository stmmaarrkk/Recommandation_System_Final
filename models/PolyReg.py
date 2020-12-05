import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
from sklearn.linear_model import LinearRegression
class PolyReg:
    def __init__(self, evaluator, deg=3):
        self.deg = deg
        self.evaluator = evaluator
        self.model = {}
        self.model["x"] = LinearRegression()
        self.model["y"] = LinearRegression()
        self.selectIndex = []
        self.inputNodesPerSeq = 0
        self.outputNodesPerSeq= 0
    def prepareData(self, data, target):
        assert data.shape[2] == 2 and target.shape[2] == 2, "The third dim should be 2"
        self.inputNodesPerSeq, self.outputNodesPerSeq = data.shape[1], target.shape[1]

        xObsv, yObsv = data[:, :, 0].copy().reshape(-1, self.inputNodesPerSeq), data[:, :, 1].copy().reshape(-1,
                                                                                                 self.inputNodesPerSeq)
        xAns, yAns = target[:, :, 0].copy().reshape(-1, self.outputNodesPerSeq), target[:, :, 1].copy().reshape(-1,
                                                                                                    self.outputNodesPerSeq)

        ##generate poly data
        poly = PolynomialFeatures(degree=self.deg)
        xObsv = poly.fit_transform(xObsv)
        yObsv = poly.fit_transform(yObsv)
        if len(self.selectIndex) == 0:
            self.getIndex()
        xObsv = xObsv[:, self.selectIndex]
        yObsv = yObsv[:, self.selectIndex]

        return xObsv, xAns, yObsv, yAns
    def fit(self, data, target):
        xObsv, xAns, yObsv, yAns = self.prepareData(data, target)

        self.model["x"].fit(xObsv, xAns)
        self.model["y"].fit(yObsv, yAns)
        xPredict = self.model["x"].predict(xObsv)
        yPredict = self.model["y"].predict(yObsv)

        rmse = self.evaluator.RMSE(xPredict, yPredict, xAns, yAns)
        fmse = self.evaluator.FMSE(xPredict, yPredict, xAns, yAns)
        print("RMSE(testing set): {:.3f}".format(rmse))
        print("FMSE(testing set): {:.3f}".format(fmse))
        #mean_squared_error(y_true, y_pred, multioutput='raw_values')
    def score(self, data, target):
        xObsv, xAns, yObsv, yAns = self.prepareData(data, target)

        xPredict = self.model["x"].predict(xObsv)
        yPredict = self.model["y"].predict(yObsv)
        rmse = self.evaluator.RMSE(xPredict, yPredict, xAns, yAns)
        fmse = self.evaluator.FMSE(xPredict, yPredict, xAns, yAns)
        print("RMSE(testing set): {:.3f}".format(rmse))
        print("FMSE(testing set): {:.3f}".format(fmse))
    def getIndex(self):
        count = 0
        q = deque()
        q.append(('', 0)) #set, level
        table = set()
        while len(q) != 0:
            cur, lv = q.popleft()
            if lv > self.deg:
                break
            if len(set(cur)) in [0, 1]:
                self.selectIndex.append(count)
            for i in range(self.inputNodesPerSeq):
                path = cur + str(i)
                if path not in table:
                    q.append((path, lv+1))

            if len(cur) == 0: #first
                count += 1
                continue

            path = "".join(sorted(cur))
            if path not in table:
                table.add(path)
                count += 1



