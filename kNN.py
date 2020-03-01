import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class kNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >=1, "k must be valid"

        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, "must fit before predict"
        assert self._X_train.shape[1] == X_predict.shape[1], "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """对每个预测向量进行结果预测"""
        distance_list = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        topK = [self._y_train[ind] for ind in np.argsort(distance_list)[:self.k]]
        votes = Counter(topK)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "kNN(K=%d)" % self.k