# 处理第三类关系
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from TheSecondRelationship import calDiscorr, single_pca
from sklearn.preprocessing import StandardScaler
from Data import splitData2xy, mergeXy2set


def dependent_x_y(x, y, th1):
    '''
    计算特征和标签的相关性，根据阈值分类
    :param x: 特征
    :param y: 标签
    :param th1: 阈值
    :return: 线性相关和非线性相关的特征序号
    '''
    linear_fe = []
    nonlinear_fe = []
    rows, cols = x.shape
    cnt = 0
    cnt1 = 0
    for i in range(0, cols):
        if (calDiscorr(np.array(x[:, i]), y) >= th1):
            a1 = i
            linear_fe.append(a1)
            cnt = cnt + 1
        elif (calDiscorr(np.array(x[:, i]), y) > 0 and calDiscorr(np.array(x[:, i]), y) < 0.7):
            zz = i
            nonlinear_fe.append(zz)
            cnt1 = cnt1 + 1
    return linear_fe, nonlinear_fe


def linear_x_y(X, y, related):
    '''
    把线性相关的特征和标签通过回归模型转化为新特征
    :param X: 特征
    :param y: 标签
    :param related: 线性相关的特征序号
    :return: 新生成的特征
    '''
    clf = Ridge(alpha=1.0)
    ans = []
    rows, cols = X.shape
    rows_val = len(related)

    # Feature matrix initialized that stores features constructed
    predicted = np.zeros((rows, len(ans)), dtype=float)

    for j in range(0, rows_val):
        rr = np.array(X[:, (int)(related[j])][:, np.newaxis])
        y_train = clf.fit(rr, y).predict(rr)[:, np.newaxis]
        predicted = np.hstack([predicted, y_train])
        # Saving constructed features finally to a file
    return predicted


def nonlinear_x_y(X, y, nonrelated):
    '''
    把非线性相关的特征进行转换
    :param X: 特征
    :param y: 标签
    :param nonrelated: 非线性相关的特征序号
    :return: 生成的新特征
    '''
    ans = []
    rows_tra, cols_tra = X.shape
    rows_val = len(nonrelated)

    # Feature matrix initialized that stores features constructed
    predicted_train = np.zeros((rows_tra, len(ans)), dtype=float)

    svr_rbf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
                          kernel_params=None)

    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    for j in range(0, rows_val):
        rr = np.array(X[:, (int)(nonrelated[j])][:, np.newaxis])

        y_train = svr_rbf.fit(rr, y).predict(rr)[:, np.newaxis]
        predicted_train = np.hstack([predicted_train, y_train])
    return predicted_train


def get_newly_feature_x_y(dataset):
    '''
    根据特征与标签的相关性生成特征
    :param dataset: 原始数据集
    :return: 新的数据集
    '''
    X, y = splitData2xy(dataset)
    related, nonrelated = dependent_x_y(X, y, 0.2)
    X_related = linear_x_y(X, y, related)
    X_nonrelated = nonlinear_x_y(X, y, nonrelated)
    r3 = np.hstack([X_related, X_nonrelated])

    scaler = StandardScaler().fit(r3)  # Normalization  & fit only on training
    X_newly = scaler.transform(r3)  # Normalized Train
    df = mergeXy2set(X_newly, y)
    return df


# def oripca_new_3nd(train, trainY, test, fold, save_to):
#     p2, p1 = get_newly_feature_x_y(train, trainY, test, fold, save_to)
#     X_train_pca, X_test_pca = pca_only(train, test, fold, save_to)
#     pca_train1 = np.hstack([p2, X_train_pca])
#     pca_test1 = np.hstack([p1, X_test_pca])
#     return pca_train1, pca_test1


if __name__ == '__main__':
    df = pd.read_csv("datasets/sonar/sonar.csv", header=None)
    print(df.head(10))
    df = get_newly_feature_x_y(df)
    print(df.shape)
