# 处理第三类关系
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from TheSecondRelationship import calDiscorr, pca_only
from sklearn.preprocessing import StandardScaler


def dependent_x_y(x, y, th1, fold, dataset_path):
    '''
    计算特征和标签之间的相关度
    :param x: 特征
    :param y: 标签
    :param th1: 阈值
    :param fold: 第几折
    :param dataset_path:数据集根据路
    :return:
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
    if os.path.exists(f'{dataset_path}linear_x_y_correlated_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}linear_x_y_correlated_{fold}.csv')
    if os.path.exists(f'{dataset_path}nonlinear_x_y_correlated_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}nonlinear_x_y_correlated_{fold}.csv')

    np.savetxt(f"{dataset_path}linear_x_y_correlated_{fold}.csv", linear_fe, delimiter=",", fmt="%s")
    np.savetxt(f"{dataset_path}nonlinear_x_y_correlated_{fold}.csv", nonlinear_fe, delimiter=",", fmt="%s")

    print("Number of linear correlated features are:")
    print(cnt)
    print("Number of non linear correlated features are:")
    print(cnt1)


# 线性特征生成
def linear_x_y(train, trainY, test, fold, dataset_path):
    '''
    :param train: 训练数据集
    :param trainY: 训练数据集标签
    :param test: 测试数据集
    :param fold: 第几折
    :return:
    '''
    clf = Ridge(alpha=1.0)
    ans = []
    rows_tra, cols_tra = train.shape
    rows_tet, cols_tet = test.shape
    dataset = pd.read_csv(f'{dataset_path}linear_x_y_correlated_{fold}.csv', header=None)
    val = dataset.as_matrix(columns=None)
    rows_val, cols_val = val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train = np.zeros((rows_tra, len(ans)), dtype=float)
    predicted_test = np.zeros((rows_tet, len(ans)), dtype=float)

    for j in range(0, rows_val):
        rr = np.array(train[:, (int)(val[j])][:, np.newaxis])
        tt = np.array(test[:, (int)(val[j])][:, np.newaxis])
        y_train = clf.fit(rr, trainY).predict(rr)[:, np.newaxis]
        y_test = clf.predict(tt)[:, np.newaxis]
        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])
    # Saving constructed features finally to a file

    if os.path.exists(f"{dataset_path}related_x_y_lineartest_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}related_x_y_lineartest_{fold}.csv")

    if os.path.exists(f'{dataset_path}related_x_y_lineartrain_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}related_x_y_lineartrain_{fold}.csv')

    with open(f"{dataset_path}related_x_y_lineartest_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_test, delimiter=",", fmt="%s")
    with open(f"{dataset_path}related_x_y_lineartrain_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_train, delimiter=",", fmt="%s")


# 非线性特征生成
def nonlinear_x_y(train, trainY, test, fold, dataset_path):
    '''
    :param train: 训练数据集
    :param test: 测试数据集
    :param fold: 第几折
    :return:
    '''
    ans = []
    rows_tra, cols_tra = train.shape
    rows_tet, cols_tet = test.shape
    dataset = pd.read_csv(f'{dataset_path}nonlinear_x_y_correlated_{fold}.csv', header=None)
    val = dataset.as_matrix(columns=None)
    rows_val, cols_val = val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train = np.zeros((rows_tra, len(ans)), dtype=float)
    predicted_test = np.zeros((rows_tet, len(ans)), dtype=float)

    svr_rbf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
                          kernel_params=None)

    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    for j in range(0, rows_val):
        rr = np.array(train[:, (int)(val[j])][:, np.newaxis])
        tt = np.array(test[:, (int)(val[j])][:, np.newaxis])

        y_train = svr_rbf.fit(rr, trainY).predict(rr)[:, np.newaxis]
        y_test = svr_rbf.fit(rr, trainY).predict(tt)[:, np.newaxis]
        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])

    if os.path.exists(f"{dataset_path}related_x_y_nonlineartest_{fold}.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}related_x_y_nonlineartest_{fold}.csv")

    if os.path.exists(
            f'{dataset_path}related_x_y_nonlineartrain_{fold}.csv'):  # Name of Ouput file generated
        os.remove(f'{dataset_path}related_x_y_nonlineartrain_{fold}.csv')

    # Saving constructed features finally to a file
    with open(f"{dataset_path}related_x_y_nonlineartest_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_test, delimiter=",")
    with open(f"{dataset_path}related_x_y_nonlineartrain_{fold}.csv", "wb") as myfile:
        np.savetxt(myfile, predicted_train, delimiter=",")


def get_newly_feature_x_y(train, trainY, test, fold, save_to):
    dependent_x_y(train, trainY, 0.2, fold, save_to)
    linear_x_y(train, trainY, test, fold, save_to)
    nonlinear_x_y(train, trainY, test, fold, save_to)

    a1 = pd.read_csv(f'{save_to}related_x_y_lineartest_{fold}.csv', header=None)  # all predicted feature files
    a2 = pd.read_csv(f'{save_to}related_x_y_lineartrain_{fold}.csv', header=None)
    a3 = pd.read_csv(f'{save_to}related_x_y_nonlineartest_{fold}.csv', header=None)
    a4 = pd.read_csv(f'{save_to}related_x_y_nonlineartrain_{fold}.csv', header=None)

    r4 = np.hstack([a2, a4])  # Train
    r3 = np.hstack([a1, a3])  # Test

    r4 = np.hstack([r4, train])
    r3 = np.hstack([r3, test])

    scaler = StandardScaler().fit(r4)  # Normalization  & fit only on training
    p2 = scaler.transform(r4)  # Normalized Train
    p1 = scaler.transform(r3)  # Normalized Test
    return p2, p1


def oripca_new_3nd(train, trainY, test, fold, save_to):
    p2, p1 = get_newly_feature_x_y(train, trainY, test, fold, save_to)
    X_train_pca, X_test_pca = pca_only(train, test, fold, save_to)
    pca_train1 = np.hstack([p2, X_train_pca])
    pca_test1 = np.hstack([p1, X_test_pca])
    return pca_train1, pca_test1
