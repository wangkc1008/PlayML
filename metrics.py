import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    """根据测试值和与测试计算模型的准确率"""
    assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to y_predict"

    return sum(y_true == y_predict) / len(y_true)

def mean_squared_error(y_true, y_predict):
    """计算均方误差"""
    assert len(y_predict) == len(y_true), "the size of y_predict must be equal to y_true"

    return np.sum((y_predict - y_true) ** 2) / len(y_predict)

def root_mean_squared_error(y_true, y_predict):
    """计算均方根误差"""
    assert len(y_predict) == len(y_true), "the size of y_predict must be equal to y_true"

    return sqrt(np.sum((y_predict - y_true) ** 2) / len(y_predict))

def mean_absolute_error(y_true, y_predict):
    """计算平均绝对误差"""
    assert len(y_predict) == len(y_true), "the size of y_predicr must be equal to y_true"

    return np.sum(np.abs(y_predict - y_true)) / len(y_predict)

def r2_score(y_true, y_predict):
    """计算R方Score"""
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)