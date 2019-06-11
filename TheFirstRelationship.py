# 处理第一类关系
import numpy as np
import pandas as pd

np.random.seed(7)


def single_cos(train, test):
    train_new = np.cos(train)
    test_new = np.cos(test)
    return train_new, test_new


def ori_cos(train, test):
    train_new = np.cos(train)
    test_new = np.cos(test)
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


def single_sin(train, test):
    train_new = np.sin(train)
    test_new = np.sin(test)
    return train_new, test_new


def ori_sin(train, test):
    train_new = np.sin(train)
    test_new = np.sin(test)
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


def single_tan(train, test):
    train_new = np.tan(train)
    test_new = np.tan(test)
    return train_new, test_new


def ori_tan(train, test):
    train_new = np.tan(train)
    test_new = np.tan(test)
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


def single_arccos(train, test):
    train_new = np.arccos(train)
    test_new = np.arccos(test)
    return train_new, test_new


def ori_arccos(train, test):
    train_new = np.arccos(train)
    test_new = np.arccos(test)
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


def single_arcsin(train, test):
    train_new = np.arcsin(train)
    test_new = np.arcsin(test)
    return train_new, test_new


def ori_arcsin(train, test):
    train_new = np.arcsin(train)
    test_new = np.arcsin(test)
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


def single_arctan(train, test):
    train_new = np.arctan(train)
    test_new = np.arctan(test)
    return train_new, test_new


def ori_arctan(train, test):
    train_new = np.arctan(train)
    test_new = np.arctan(test)
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


def single_square(train, test):
    train_new = train ** 2
    test_new = test ** 2
    return train_new, test_new


def ori_square(train, test):
    train_new = train ** 2
    test_new = test ** 2
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


# 归一化
def single_normalizetion(train, test):
    train_new = (train - train.mean()) / train.std()
    test_new = (test - test.mean()) / test.std()
    return train_new, test_new


def ori_normalizetion(train, test):
    train_new = (train - train.mean()) / train.std()
    test_new = (test - test.mean()) / test.std()
    train_new = np.hstack([train, train_new])
    test_new = np.hstack([test, test_new])
    return train_new, test_new


# 离散化
def single_discretization(train, test):
    train_new = pd.cut(train, 10, precision=2, labels=False)
    test_new = pd.cut(test, 10, precision=2, labels=False)
    return train_new, test_new


# one-hot
def single_onehot(npArray):
    pass


if __name__ == '__main__':
    data = np.random.rand(20)
    print(data)
    print(single_discretization(data))
