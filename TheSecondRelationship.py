# 处理第二类关系
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats.stats import pearsonr
import os
from Selection import selectByIG
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from Data import splitData2xy, mergeXy2set
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor

np.random.seed(7)


# 计算相关系数


# 计算相关距离
def calDiscorr(X, Y):
    '''
    :param X:特征
    :param Y: 标签
    :return: 距离矩阵
    '''
    X = np.atleast_1d(X)  # 视为一维
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)  # 视为二维
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


# 得到线性和非线性相关特征
def dependent(x, th1):
    '''
    :param x: 特征集
    :param th1: 阈值
    :param fold: 第几折
    :return:  ans:线性相关特征对  ans1:非线性
    '''
    related = []
    nonrelated = []
    m, n = x.shape
    cnt = 0
    cnt1 = 0
    for i in range(0, n):
        for j in range(0, n):
            if (i != j):
                a, b = pearsonr(x[:, i][:, np.newaxis], x[:, j][:, np.newaxis])
                if (calDiscorr(np.array(x[:, i]), np.array(x[:, j])) >= th1):
                    a1 = i, j
                    related.append(a1)
                    cnt = cnt + 1
                elif (calDiscorr(np.array(x[:, i]), np.array(x[:, j])) > 0 and calDiscorr(np.array(x[:, i]),
                                                                                          np.array(x[:, j])) < 0.7):
                    zz = i, j
                    nonrelated.append(zz)
                    cnt1 = cnt1 + 1
    return related, nonrelated


# 线性特征生成
def linear(X, related):
    '''
    :param TR: 训练数据集
    :param TST: 测试数据集
    :param fold: 第几折
    :return:
    '''
    # clf = Ridge(alpha=1.0)
    # clf = LinearRegression()
    clf = RandomForestRegressor()
    ans = []
    rows, cols = X.shape
    aa = len(related)
    predicted = np.zeros((rows, len(ans)), dtype=float)
    predicted_error = np.zeros((rows, len(ans)), dtype=float)

    for j in range(0, aa):
        rr, ss = np.array(X[:, (int)(related[j][0])][:, np.newaxis]), np.array(X[:, (int)(related[j][1])])
        y_train = clf.fit(rr, ss).predict(rr)[:, np.newaxis]
        predicted = np.hstack([predicted, y_train])

        dd = ss[:, np.newaxis]
        diff_train = (dd - y_train)
        predicted_error = np.hstack([predicted_error, diff_train])
    predicted_final = np.hstack([predicted, predicted_error])

    return predicted_final


# 非线性特征生成
def nonlinear(X, nonrelated):
    '''
    :param TR: 训练数据集
    :param TST: 测试数据集
    :param fold: 第几折
    :return:
    '''
    ans = []
    rows, cols = X.shape
    aa = len(nonrelated)
    predicted = np.zeros((rows, len(ans)), dtype=float)
    predicted_error = np.zeros((rows, len(ans)), dtype=float)
    # svr_rbf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
    #                       kernel_params=None)
    # svr_rbf = SVR()
    svr_rbf = AdaBoostRegressor()
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    for j in range(0, aa):
        rr, ss = np.array(X[:, (int)(nonrelated[j][0])][:, np.newaxis]), np.array(X[:, (int)(nonrelated[j][1])])
        y_train = svr_rbf.fit(rr, ss).predict(rr)[:, np.newaxis]
        predicted = np.hstack([predicted, y_train])
        dd = ss[:, np.newaxis]
        diff_train = (dd - y_train)
        predicted_error = np.hstack([predicted_error, diff_train])

    predicted_final = np.hstack([predicted, predicted_error])
    return predicted_final


def ori_pca(dataset):
    '''
    特征经过pca以后再加上原始特征
    :param dataset: 原始数据集
    :return: 新数据集
    '''
    X, y = splitData2xy(dataset)
    rows, cols = X.shape
    ipca = PCA(n_components=min(60, cols))
    ipca.fit(X)
    X_new = ipca.transform(X)

    scaler = StandardScaler().fit(X_new)
    X_sca = scaler.transform(X)
    X_new = np.hstack([X_new, X_sca])
    df = mergeXy2set(X_new, y)
    return df


def single_pca(dataset):
    '''
    特征通过PCA进行转化
    :param dataset: 原始数据集
    :return: 转化后的数据集
    '''
    X, y = splitData2xy(dataset)
    rows, cols = X.shape
    ipca = PCA(n_components=min(60, cols))
    ipca.fit(X)
    X_new = ipca.transform(X)
    df = mergeXy2set(X_new, y)
    return df


# PCA+ori然后再ig
def pca_ori_ig(X_train, X_test, Y_train, fold, save_to):
    '''
    先对原始特征进行pca加上原始特征然后再ig
    :param X_train:
    :param X_test:
    :param Y_train:
    :param fold:
    :param save_to:
    :return:
    '''
    X_train_pca, X_test_pca = pca_operator(X_train, X_test, fold, save_to)
    X_train_pca, X_test_pca = selectByIG(X_train_pca, X_test_pca, Y_train, save_to)
    return X_train_pca, X_test_pca


# 对原始特征进行PCA,然后加上newly特征
def oripca_new(X_train, X_test, fold, save_to, p2, p1):
    '''
    先对原始特征进行pca,然后加上newly特征
    :param X_train:
    :param X_test:
    :param fold:
    :param save_to:
    :param p2:
    :param p1:
    :return:
    '''
    X_train_pca, X_test_pca = pca_only(X_train, X_test, fold, save_to)
    pca_train1 = np.hstack([p2, X_train_pca])
    pca_test1 = np.hstack([p1, X_test_pca])
    return pca_train1, pca_test1


# 对原始特征进行PCA,加上原始特征,然后加上newly特征
def oripca_ori_new(X_train, X_test, fold, save_to, p2, p1):
    '''
    先对原始特征进行pca,加上原始特征，然后加上newly特征
    :param X_train:
    :param X_test:
    :param fold:
    :param save_to:
    :param p2:
    :param p1:
    :return:
    '''
    X_train_pca, X_test_pca = pca_operator(X_train, X_test, fold, save_to)
    pca_train1 = np.hstack([p2, X_train_pca])
    pca_test1 = np.hstack([p1, X_test_pca])
    return pca_train1, pca_test1


# PCA+ORI->PCA+NEW
def double_oripca_ori_new(X_train, X_test, fold, save_to, p2, p1):
    pca_train1, pca_test1 = pca_operator(X_train, X_test, fold, save_to)
    pca_train1, pca_test1 = pca_only(pca_train1, pca_test1, fold, save_to)
    pca_train1 = np.hstack([p2, pca_train1])
    pca_test1 = np.hstack([p1, pca_test1])
    return pca_train1, pca_test1


# PCA+ORI->ig->pca+new
def oripca_ori_ig_pca_ori_new(X_train, X_test, Y_train, fold, save_to, p2, p1):
    X_train_pca, X_test_pca = pca_ori_ig(X_train, X_test, Y_train, fold, save_to)
    X_train_pca, X_test_pca = pca_only(X_train_pca, X_test_pca, fold, save_to)
    pca_train1 = np.hstack([p2, X_train_pca])
    pca_test1 = np.hstack([p1, X_test_pca])
    return pca_train1, pca_test1


def newly_feature(dataset):
    '''
    获得线性和非线性生成特征
    :param train: 训练
    :param test: 测试
    :return:
    '''
    X, y = splitData2xy(dataset)
    related, nonrelated = dependent(X, 0.7)
    X_linear = linear(X, related)
    X_nonlinear = nonlinear(X, nonrelated)

    X_newly = np.hstack([X_linear, X_nonlinear])

    scaler = StandardScaler().fit(X_newly)  # Normalization  & fit only on training
    X_newly_sca = scaler.transform(X_newly)
    df = mergeXy2set(X_newly_sca, y)
    return df


def ori_newly(dataset):
    '''
    获得线性和非线性生成特征
    :param train: 训练
    :param test: 测试
    :return:
    '''
    X, y = splitData2xy(dataset)
    related, nonrelated = dependent(X, 0.7)
    X_linear = linear(X, related)
    X_nonlinear = nonlinear(X, nonrelated)

    X_newly = np.hstack([X_linear, X_nonlinear])

    scaler = StandardScaler().fit(X_newly)  # Normalization  & fit only on training
    X_newly_sca = scaler.transform(X_newly)

    X_newly_sca_ori = np.hstack([X, X_newly_sca])
    df = mergeXy2set(X_newly_sca_ori, y)
    return df


def get_select_newly(train, test, trainY, dataset_path):
    '''
    从新生成的特征中选择
    :param train: 训练
    :param test: 测试
    :param trainY: 训练标签
    :return:
    '''
    stable(train, test, trainY, dataset_path)
    f1 = pd.read_csv(f'{dataset_path}ensemble_trainfeatures.csv', header=None)
    f2 = pd.read_csv(f'{dataset_path}ensemble_testfeatures.csv', header=None)

    scaler = StandardScaler().fit(f1)
    e_f1 = scaler.transform(f1)
    e_f2 = scaler.transform(f2)
    return e_f1, e_f2, f1, f2


def get_ensemble_feature(ori_train, new_train, ori_test, new_test):
    '''
    把原始特征和新特征组合
    :param ori_train: 训练集原始特征
    :param new_train: 训练集生成特征
    :param ori_test: 测试集原始特征
    :param new_test: 测试集生成特征
    :return:
    '''
    x1X = np.hstack(
        [ori_test, new_test])  # original test features, selected by IG, f2 is feature space after ensemble selection.
    x2X = np.hstack([ori_train, new_train])

    scaler = StandardScaler().fit(x2X)  # Again normalization of the complete combined feature pool
    x2 = scaler.transform(x2X)  # note - when features need to be merged with R2R, we need to do normalization.
    x1 = scaler.transform(x1X)

    y1Y = np.hstack([ori_test, new_test])
    y2Y = np.hstack([ori_train, new_train])

    scaler = StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
    y2 = scaler.transform(y2Y)  # note - when features need to be merged with R2R, we need to do normalization.
    y1 = scaler.transform(y1Y)

    return x2, x1, y2, y1


def get_stable_feature(ori_ig_train, ori_ig_test, dataset_path):
    '''
    把筛选后的原始特征和筛选后的生成特征组合
    :param ori_ig_train: 筛选后的原始训练集
    :param ori_ig_test: 筛选后的原始测试集
    :return:
    '''
    st_f1 = pd.read_csv(f'{dataset_path}stable_trainfeatures.csv', header=None)
    st_f2 = pd.read_csv(f'{dataset_path}stable_testfeatures.csv', header=None)

    st_x1X = np.hstack([ori_ig_test,
                        st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
    st_x2X = np.hstack([ori_ig_train, st_f1])

    scaler = StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
    st_x2 = scaler.transform(st_x2X)  # note - when features need to be merged with R2R, we need to do normalization.
    st_x1 = scaler.transform(st_x1X)
    return st_x2, st_x1


if __name__ == '__main__':
    df = pd.read_csv("datasets/sonar/sonar.csv", header=None)
    print(df.head(10))
    df = get_newly_feature(df)
    print(df.shape)
