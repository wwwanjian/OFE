# 特征选择
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import RandomizedLasso
import numpy as np
import os

np.random.seed(7)


# 特征筛选  通过IG 得到筛选特征列表
def selectByIG(ress, test, labels, dataset_path):
    '''
    :param ress:训练数据集
    :param test:测试数据集
    :param labels: 训练数据集标签
    :return:
    '''
    x, y = ress.shape
    names = np.arange(y)

    ress_new = SelectKBest(mutual_info_classif, k='all')  # 计算IG
    ress_new.fit_transform(ress, labels)

    # print "Features sorted by their scores according to the scoring function - mutual information gain:"
    original_features = sorted(zip(map(lambda x: round(x, 4), ress_new.scores_),
                                   names), reverse=True)

    finale = []  # 选择后的特征
    for i in range(0, len(original_features)):
        r, s = original_features[i]
        if (r > 0):  # This is eta-o
            finale.append(s)

    print("Selected features after O + IG:")
    print(len(finale))
    dataset1 = ress[:, finale]
    dataset3 = test[:, finale]

    # if os.path.exists(f"{dataset_path}original_ig_testfeatures.csv"):  # Name of Ouput file generated
    #     os.remove(f"{dataset_path}original_ig_testfeatures.csv")
    # if os.path.exists(f"{dataset_path}original_ig_trainfeatures.csv"):  # Name of Ouput file generated
    #     os.remove(f"{dataset_path}original_ig_trainfeatures.csv")
    #
    # with open(f"{dataset_path}original_ig_testfeatures.csv", "wb") as myfile:
    #     np.savetxt(myfile, dataset3, delimiter=",", fmt="%s")
    # with open(f"{dataset_path}original_ig_trainfeatures.csv", "wb") as myfile:
    #     np.savetxt(myfile, dataset1, delimiter=",", fmt="%s")
    return dataset1, dataset3


# 从新生成的特征进行选择
def stable(ress, test, labels, dataset_path):  # ress is training data
    '''
    :param ress:训练数据集
    :param test: 测试数据集
    :param labels: 训练数据集标签
    :return:
    '''
    x, y = ress.shape
    names = np.arange(y)
    rlasso = RandomizedLasso()
    rlasso.fit(ress, labels)

    # print "Features sorted by their scores according to the stability scoring function"
    val = sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                     names), reverse=True)

    print("len of val")  # newly constructed features
    print(len(val))

    finale = []
    for i in range(0, len(val)):
        r, s = val[i]  # 'r' represents scores, 's' represents column name
        if (r > 0.1):  # This is eta for stability selection
            finale.append(s)
            # finale.append(s)

    print("Total features after stability selection:")
    print(len(finale))  # finale stores col names - 2nd, 4th etc of stable features.

    dataset1 = ress[:, finale]
    dataset3 = test[:, finale]

    if os.path.exists(f"{dataset_path}stable_testfeatures.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}stable_testfeatures.csv")
    if os.path.exists(f"{dataset_path}stable_trainfeatures.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}stable_trainfeatures.csv")

    with open(f"{dataset_path}stable_testfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset3, delimiter=",", fmt="%s")
    with open(f"{dataset_path}stable_trainfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset1, delimiter=",", fmt="%s")

    # -----------------------------------------------------------------------------------
    # check the inter-feature dependence - 2nd phase of ensemble

    ress_new = SelectKBest(mutual_info_classif, k='all')
    ress_new.fit_transform(ress[:, finale], labels)

    # print "Features sorted by their scores according to the scoring function - mutual information gain:"
    feats = sorted(zip(map(lambda x: round(x, 4), ress_new.scores_),
                       names), reverse=True)

    ensemble_finale = []
    for i in range(0, len(feats)):
        r, s = feats[i]
        if (r > 0):  # This is eta-o
            ensemble_finale.append(s)

    print("Total features after 2 phase selection:")
    print(len(ensemble_finale))  # ensemble_finale stores col names further pruned in the 2nd phase of feature selection

    dataset2 = ress[:, ensemble_finale]
    dataset4 = test[:, ensemble_finale]

    if os.path.exists(f"{dataset_path}ensemble_testfeatures.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}ensemble_testfeatures.csv")
    if os.path.exists(f"{dataset_path}ensemble_trainfeatures.csv"):  # Name of Ouput file generated
        os.remove(f"{dataset_path}ensemble_trainfeatures.csv")

    with open(f"{dataset_path}ensemble_testfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset4, delimiter=",", fmt="%s")
    with open(f"{dataset_path}ensemble_trainfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset2, delimiter=",", fmt="%s")
