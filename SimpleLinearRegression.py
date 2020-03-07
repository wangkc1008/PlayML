import numpy as np

class SimpleLinearRegression1():

    def __init__(self):
        """初始化 Single Linear Regression 的方法"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练集x_train y_train训练得到SimpleLinearRegression1的模型"""
        assert x_train.ndim == 1, "the Simple Linear Regression can only solve single feature training data"
        assert len(x_train) == len(y_train), "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        up = 0.0
        down = 0.0
        for x_i, y_i in zip(x_train, y_train):
            up += (x_i - x_mean) * (y_i - y_mean)
            down += (x_i - x_mean) ** 2

        self.a_ = up / down
        self.b_ = y_mean - self.a_ * x_mean

    def predict(self, x_test):
        """通过训练好的SimpleLinearRegression1模型计算得到x_test的预测值"""
        assert x_test.ndim == 1, "the Simple Linear Regression can only solve single feature test data"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict"

        y_predict = [self._predict(x) for x in x_test]
        return np.array(y_predict)

    def _predict(self, x):
        """对给定的单个待预测数据x，返回预测结果值"""
        return self.a_ * x + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"