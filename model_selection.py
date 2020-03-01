import numpy as np

def train_test_split(X, y, test_size = 0.2, random_state = None):
    """将数据X,y按照测试比例进行分割"""
    assert X.shape[0] == y.shape[0], "the size of X must be equal to y"
    assert 0.0 <= test_size <= 1.0, "the test_size must be valid"

    if random_state:
        np.random.seed(random_state)

    shuffled_indexes = np.random.permutation(len(X))
    index = int(test_size * len(X))

    test_indexes = shuffled_indexes[:index]
    train_indexes = shuffled_indexes[index:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test