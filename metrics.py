import numpy as np

def accuracy_score(y_test, y_predict):
    """根据测试值和与测试计算模型的准确率"""
    assert y_test.shape[0] == y_predict.shape[0], "the size of y_test must be equal to y_predict"

    return sum(y_test == y_predict) / len(y_test)