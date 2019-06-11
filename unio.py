# 联合特征
from TheFirstRelationship import single_sin, ori_arccos
from TheSecondRelationship import pca_only, oripca_ori_new
import numpy as np


def _8_Plus_21(X_train, X_test, fold, save_to):
    X_train_pca, X_test_pca = pca_only(X_train, X_test, fold, save_to)
    X_train_sin, X_test_sin = single_sin(X_train, X_test)
    train = np.hstack([X_train_pca, X_train_sin])
    test = np.hstack([X_test_pca, X_test_sin])
    return train, test


def _12_plus_30(X_train, X_test, fold, save_to, p2, p1):
    X_train_pca, X_test_pca = oripca_ori_new(X_train, X_test, fold, save_to, p2, p1)
    X_train_sin, X_test_sin = ori_arccos(X_train, X_test)
    train = np.hstack([X_train_pca, X_train_sin])
    test = np.hstack([X_test_pca, X_test_sin])
    return train, test
