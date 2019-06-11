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
def dependent(x, th1, fold, dataset_path):
    '''
    :param x: 特征集
    :param th1: 阈值
    :param fold: 第几折
    :return:  ans:线性相关特征对  ans1:非线性
    '''
    ans = []
    ans1 = []
    m, n = x.shape
    cnt = 0
    cnt1 = 0
    for i in range(0, n):
        for j in range(0, n):
            if (i != j):
                a, b = pearsonr(x[:, i][:, np.newaxis], x[:, j][:, np.newaxis])
                if (calDiscorr(np.array(x[:, i]), np.array(x[:, j])) >= th1):
                    a1 = i, j
                    ans.append(a1)
                    cnt = cnt + 1
                elif (calDiscorr(np.array(x[:, i]), np.array(x[:, j])) > 0 and calDiscorr(np.array(x[:, i]),
                                                                                          np.array(x[:, j])) < 0.7):
                    zz = i, j
                    ans1.append(zz)
                    cnt1 = cnt1 + 1
    if os.path.exists(f'{dataset_path}linear_correlated_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}linear_correlated_{fold}.csv')
    if os.path.exists(f'{dataset_path}nonlinear_correlated_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}nonlinear_correlated_{fold}.csv')

    np.savetxt(f"{dataset_path}linear_correlated_{fold}.csv", ans, delimiter=",", fmt="%s")
    np.savetxt(f"{dataset_path}nonlinear_correlated_{fold}.csv", ans1, delimiter=",", fmt="%s")

    print("Number of linear correlated features are:")
    print(cnt)
    print("Number of non linear correlated features are:")
    print(cnt1)


# 线性特征生成
def linear(TR, TST, fold, dataset_path):
    '''
    :param TR: 训练数据集
    :param TST: 测试数据集
    :param fold: 第几折
    :return:
    '''
    clf = Ridge(alpha=1.0)
    # clf = LinearRegression()
    # clf = RandomForestRegressor()
    ans = []
    a, b = TR.shape
    c, d = TST.shape
    dataset = pd.read_csv(f'{dataset_path}linear_correlated_{fold}.csv', header=None)
    val = dataset.as_matrix(columns=None)
    aa, bb = val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train = np.zeros((a, len(ans)), dtype=float)
    predicted_test = np.zeros((c, len(ans)), dtype=float)
    predicted_train_error = np.zeros((a, len(ans)), dtype=float)
    predicted_test_error = np.zeros((c, len(ans)), dtype=float)

    for j in range(0, aa):
        rr, ss = np.array(TR[:, (int)(val[j][0])][:, np.newaxis]), np.array(TR[:, (int)(val[j][1])])
        tt, uu = np.array(TST[:, (int)(val[j][0])][:, np.newaxis]), np.array(TST[:, (int)(val[j][1])])
        y_train = clf.fit(rr, ss).predict(rr)[:, np.newaxis]
        y_test = clf.fit(rr, ss).predict(tt)[:, np.newaxis]
        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])

        dd = ss[:, np.newaxis]
        ee = uu[:, np.newaxis]
        diff_train = (dd - y_train)
        diff_test = (ee - y_test)
        predicted_train_error = np.hstack([predicted_train_error, diff_train])
        predicted_test_error = np.hstack([predicted_test_error, diff_test])
    predicted_train_final = np.hstack([predicted_train, predicted_train_error])
    predicted_test_final = np.hstack([predicted_test, predicted_test_error])
    # Saving constructed features finally to a file

    if os.path.exists(f"{dataset_path}related_lineartest_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}related_lineartest_{fold}.csv")

    if os.path.exists(f'{dataset_path}related_lineartrain_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}related_lineartrain_{fold}.csv')

    with open(f"{dataset_path}related_lineartest_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_test_final, delimiter=",", fmt="%s")
    with open(f"{dataset_path}related_lineartrain_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_train_final, delimiter=",", fmt="%s")

    return predicted_train_final, predicted_test_final


# 非线性特征生成
def nonlinear(TR, TST, fold, dataset_path):
    '''
    :param TR: 训练数据集
    :param TST: 测试数据集
    :param fold: 第几折
    :return:
    '''
    ans = []
    a, b = TR.shape
    c, d = TST.shape
    dataset = pd.read_csv(f'{dataset_path}nonlinear_correlated_{fold}.csv', header=None)
    val = dataset.as_matrix(columns=None)
    aa, bb = val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train = np.zeros((a, len(ans)), dtype=float)
    predicted_test = np.zeros((c, len(ans)), dtype=float)

    predicted_train_error = np.zeros((a, len(ans)), dtype=float)
    predicted_test_error = np.zeros((c, len(ans)), dtype=float)

    svr_rbf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
                          kernel_params=None)
    # svr_rbf = SVR()
    # svr_rbf = AdaBoostRegressor()

    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    for j in range(0, aa):
        rr, ss = np.array(TR[:, (int)(val[j][0])][:, np.newaxis]), np.array(TR[:, (int)(val[j][1])])
        tt, uu = np.array(TST[:, (int)(val[j][0])][:, np.newaxis]), np.array(TST[:, (int)(val[j][1])])

        y_train = svr_rbf.fit(rr, ss).predict(rr)[:, np.newaxis]
        y_test = svr_rbf.fit(rr, ss).predict(tt)[:, np.newaxis]
        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])

        dd = ss[:, np.newaxis]
        ee = uu[:, np.newaxis]
        diff_train = (dd - y_train)
        diff_test = (ee - y_test)

        predicted_train_error = np.hstack([predicted_train_error, diff_train])
        predicted_test_error = np.hstack([predicted_test_error, diff_test])

    predicted_train_final = np.hstack([predicted_train, predicted_train_error])
    predicted_test_final = np.hstack([predicted_test, predicted_test_error])

    if os.path.exists(f"{dataset_path}related_nonlineartest_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}related_nonlineartest_{fold}.csv")

    if os.path.exists(
            f'{dataset_path}related_nonlineartrain_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}related_nonlineartrain_{fold}.csv')

    # Saving constructed features finally to a file
    with open(f"{dataset_path}related_nonlineartest_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_test_final, delimiter=",")
    with open(f"{dataset_path}related_nonlineartrain_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_train_final, delimiter=",")

    return predicted_train_final, predicted_test_final


# PCA 生成特征+origin特征
def pca_operator(X_train, X_test, fold, save_to):
    '''
    通过PCA生成特征在和原始特征拼接起来
    :param X_train:
    :param X_test:
    :param fold:
    :param save_to:
    :return:
    '''
    rows, cols = X_train.shape
    ipca = PCA(n_components=min(60, cols))
    ipca.fit(X_train)
    X_train_pca = ipca.transform(X_train)
    X_test_pca = ipca.transform(X_test)

    scaler = StandardScaler().fit(X_train_pca)
    X_train_sca = scaler.transform(X_train)
    X_test_sca = scaler.transform(X_test)

    X_train_pca = np.hstack([X_train_pca, X_train_sca])
    X_test_pca = np.hstack([X_test_pca, X_test_sca])

    if os.path.exists(f"{save_to}pca_train_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{save_to}pca_train_{fold}.csv")
    if os.path.exists(f"{save_to}pca_test_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{save_to}pca_test_{fold}.csv")

    # Saving constructed features finally to a file
    with open(f"{save_to}pca_train_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, X_train_pca, delimiter=",")
    with open(f"{save_to}pca_test_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, X_test_pca, delimiter=",")

    return X_train_pca, X_test_pca


# 单独PCA特征
def pca_only(X_train, X_test, fold, save_to):
    '''
    通过PCA生成特征
    :param X_train:
    :param X_test:
    :param fold:
    :param save_to:
    :return:
    '''
    rows, cols = X_train.shape
    ipca = PCA(n_components=min(60, cols))
    ipca.fit(X_train)
    X_train_pca = ipca.transform(X_train)
    X_test_pca = ipca.transform(X_test)

    if os.path.exists(f"{save_to}pca_only_train_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{save_to}pca_only_train_{fold}.csv")
    if os.path.exists(f"{save_to}pca_only_test_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{save_to}pca_only_test_{fold}.csv")

    # Saving constructed features finally to a file
    with open(f"{save_to}pca_only_train_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, X_train_pca, delimiter=",")
    with open(f"{save_to}pca_only_test_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, X_test_pca, delimiter=",")

    return X_train_pca, X_test_pca


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