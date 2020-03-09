import numpy as np
from math import sqrt

def accuracy_score(y_test, y_predict):
    """根据测试值和与测试计算模型的准确率"""
    assert y_test.shape[0] == y_predict.shape[0], "the size of y_test must be equal to y_predict"

    return sum(y_test == y_predict) / len(y_test)

def mean_squared_error(y_predict, y_test):
    """计算均方误差"""
    assert len(y_predict) == len(y_test), "the size of y_predict must be equal to y_test"

    return np.sum((y_predict - y_test) ** 2) / len(y_predict)

def root_mean_squared_error(y_predict, y_test):
    """计算均方根误差"""
    assert len(y_predict) == len(y_test), "the size of y_predict must be equal to y_test"

    return sqrt(np.sum((y_predict - y_test) ** 2) / len(y_predict))

def mean_absolute_error(y_predict, y_test):
    """计算平均绝对误差"""
    assert len(y_predict) == len(y_test), "the size of y_predicr must be equal to y_test"

    return np.sum(np.abs(y_predict - y_test)) / len(y_predict)