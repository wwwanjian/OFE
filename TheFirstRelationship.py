# 处理第一类关系
import numpy as np
import pandas as pd
from Data import splitData2xy, mergeXy2set

np.random.seed(7)


def single_cos(dataset):
    '''
    特征经过cos转换
    :param dataset: 原始数据集
    :return: 转换后的数据集
    '''
    X, y = splitData2xy(dataset)
    X_new = np.cos(X)
    df = mergeXy2set(X_new, y)
    return df


def ori_cos(dataset):
    '''
    特征经过cos转化后加上原始特征
    :param dataset:
    :return:
    '''
    X, y = splitData2xy(dataset)
    X_new = np.cos(X)
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


def single_sin(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.sin(X)
    df = mergeXy2set(X_new, y)
    return df


def ori_sin(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.sin(X)
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


def single_tan(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.tan(X)
    df = mergeXy2set(X_new, y)
    return df


def ori_tan(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.tan(X)
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


def single_arccos(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.arccos(X)
    df = mergeXy2set(X_new, y)
    return df


def ori_arccos(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.arccos(X)
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


def single_arcsin(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.arcsin(X)
    df = mergeXy2set(X_new, y)
    return df


def ori_arcsin(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.arcsin(X)
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


def single_arctan(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.arctan(X)
    df = mergeXy2set(X_new, y)
    return df


def ori_arctan(dataset):
    X, y = splitData2xy(dataset)
    X_new = np.arctan(X)
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


def single_square(dataset):
    X, y = splitData2xy(dataset)
    X_new = X ** 2
    df = mergeXy2set(X_new, y)
    return df


def ori_square(dataset):
    X, y = splitData2xy(dataset)
    X_new = X ** 2
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


# 归一化
def single_normalizetion(dataset):
    X, y = splitData2xy(dataset)
    X_new = (X - X.mean()) / X.std()
    df = mergeXy2set(X_new, y)
    return df


def ori_normalizetion(dataset):
    X, y = splitData2xy(dataset)
    X_new = (X - X.mean()) / X.std()
    X_new = np.hstack([X, X_new])
    df = mergeXy2set(X_new, y)
    return df


# 离散化
def single_discretization(dataset):
    X, y = splitData2xy(dataset)
    X_new = pd.cut(X, 10, precision=2, labels=False)
    df = mergeXy2set(X_new, y)
    return df


# one-hot
def single_onehot(npArray):
    pass


if __name__ == '__main__':
    df = pd.read_csv("datasets/sonar/sonar.csv", header=None)
    print(df.head(10))
    nes = ori_normalizetion(df)
    print(nes)
    print(nes.shape)
