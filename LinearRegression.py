import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None #斜率
        self.intercept_ = None #截距
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练集X_train, y_train使用多元线性回归法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones(shape=(X_train.shape[0], 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X_train, y_train, initial_theta, eta, n_iters=1e4, epslion=1e-8):
        """根据训练集X_train, y_train使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """计算损失函数"""
            try:
                return np.sum((X_b.dot(theta) - y) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            """计算梯度值"""
            res = np.empty(len(theta))
            res[0] = np.sum((X_b.dot(theta) - y))
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])

            return res * 2 / len(X_b)

        def gradient_decent(X_b, y, initial_theta, eta, n_iters=1e4, epslion=1e-8):
            """计算梯度下降法中的theta"""
            theta = initial_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta -= eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y))):
                    break

                i_iter += 1

            return theta

        X_b = np.hstack([np.ones(shape=(len(X_train), 1)), X_train])
        initial_theta = np.zeros(shape=(X_b.shape[1]))

        self._theta = gradient_decent(X_b, y_train, initial_theta, eta)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_test):
        """根据训练到的Linear Regression模型对测试集X_test进行预测"""
        assert X_test.shape[1] == len(self.coef_), "the size of X_test must be equal to the size of the coef"
        assert self.coef_ is not None or self.intercept_ is not None, "must fit before predict"

        X_b = np.hstack([np.ones(shape=(X_test.shape[0], 1)), X_test])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据X_test, y_test计算r2_score"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"