import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None #斜率
        self.interception_ = None #截距
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones(shape=(X_train.shape[0], 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predict(self, X_test):
        """根据训练到的Linear Regression模型对测试集X_test进行预测"""
        assert X_test.shape[1] == len(self.coef_), "the size of X_test must be equal to the size of the coef"
        assert self.coef_ is not None or self.interception_ is not None, "must fit before predict"

        X_b = np.hstack([np.ones(shape=(X_test.shape[0], 1)), X_test])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据X_test, y_test计算r2_score"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"