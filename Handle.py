# 核心处理文件
import pandas as pd
from Data import shuffle, splitDataToCouples
import numpy as np
from Selection import selectByIG, stable
from sklearn.svm import SVC
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from TheSecondRelationship import newly_feature, ori_newly, \
    ori_pca, pca_ori_ig, oripca_new, oripca_ori_new, double_oripca_ori_new, oripca_ori_ig_pca_ori_new
from TheFirstRelationship import single_sin, single_cos, single_tan, single_arcsin, single_arccos, single_arctan, \
    single_square, single_discretization, single_normalizetion, \
    ori_sin, ori_cos, ori_tan, ori_arccos, ori_arcsin, ori_arctan, ori_normalizetion, ori_square
from TheThirdRelationship import get_newly_feature_x_y
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from unio import A_plus_B
from models import single_train, cross_val_train

np.random.seed(7)


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


def train_for_one(models, names, train, trainY, test, testY, score):
    for i in range(0, len(models)):
        models[i].fit(train, trainY)
        y_out = models[i].predict(test)
        print(models[i].score(test, testY), " ..... ", names[i])
        score[names[i]] += models[i].score(test, testY)


def train_and_prediction(models, train, trainY, test, testY, names, original, ori_ig_train, ori_ig_test, orig_ig, p2,
                         p1, new, e_f1, e_f2, new_fs, y2, y1, supplement, x2, x1, supplement_ig, st_x2, st_x1,
                         stable_ig):
    train_for_one(models, names, train, trainY, test, testY, original)
    train_for_one(models, names, ori_ig_train, trainY, ori_ig_test, testY, orig_ig)
    train_for_one(models, names, p2, trainY, p1, testY, new)
    train_for_one(models, names, e_f1, trainY, e_f2, testY, new_fs)
    train_for_one(models, names, y2, trainY, y1, testY, supplement)
    train_for_one(models, names, x2, trainY, x1, testY, supplement_ig)
    train_for_one(models, names, st_x2, trainY, st_x1, testY, stable_ig)


def main():
    df = pd.read_csv(dataset_path, header=None)
    df = shuffle(df)
    data = df.sample(frac=1)
    # 划分数据集
    train1, train1Y, test1, test1Y, \
    train2, train2Y, test2, test2Y, \
    train3, train3Y, test3, test3Y, \
    train4, train4Y, test4, test4Y, \
    train5, train5Y, test5, test5Y = splitDataToCouples(data)
    # 统计结果
    original = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, \
                'AdaBoost': 0, 'Neural Network': 0, 'Decision Tree': 0}
    orig_ig = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, \
               'AdaBoost': 0, 'Neural Network': 0, 'Decision Tree': 0}
    new = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, 'AdaBoost': 0, \
           'Neural Network': 0, 'Decision Tree': 0}
    new_fs = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, 'AdaBoost': 0, \
              'Neural Network': 0, 'Decision Tree': 0}
    supplement = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, \
                  'AdaBoost': 0, 'Neural Network': 0, 'Decision Tree': 0}
    supplement_ig = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, \
                     'AdaBoost': 0, 'Neural Network': 0, 'Decision Tree': 0}
    stable_ig = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, \
                 'AdaBoost': 0, 'Neural Network': 0, 'Decision Tree': 0}
    pca_with_ori = {'kNN': 0, 'Logistic Regression': 0, 'Linear SVM': 0, 'Poly SVM': 0, 'Random Forest': 0, \
                    'AdaBoost': 0, 'Neural Network': 0, 'Decision Tree': 0}

    # 构建模型
    print("..........................................................................................")
    names = ['kNN', 'Logistic Regression', 'Linear SVM', 'Poly SVM', 'Random Forest', 'AdaBoost', 'Neural Network',
             'Decision Tree']
    models = [KNeighborsClassifier(), LogisticRegression(), svm.LinearSVC(), SVC(C=1.0, kernel='poly'),
              RandomForestClassifier(), AdaBoostClassifier(), MLPClassifier(), tree.DecisionTreeClassifier()]

    # 获取各种特征集
    original_ig_train1, original_ig_test1 = get_ori_ig(train1, test1, train1Y, save_to)
    p2, p1 = get_newly_feature(original_ig_train1, original_ig_test1, 1, save_to)
    # e_f1, e_f2, f1, f2 = get_select_newly(p2, p1, train1Y, save_to)
    # x2, x1, y2, y1 = get_ensemble_feature(train1, f1, test1, f2)
    # st_x2, st_x1 = get_stable_feature(original_ig_train1, original_ig_test1, save_to)
    pca_train, pca_test = oripca_ori_new(train1, test1, 1, save_to, p2, p1)
    # single_train, single_test = single_normalizetion(train1, test1)
    # train_3rd, test_2nd = get_newly_feature_x_y(train1, train1Y, test1, 1, save_to)
    # train = np.hstack([pca_train, single_train, train_3rd])
    # test = np.hstack([pca_test, single_test, test_2nd])
    # train_unio, test_unio = _12_plus_30(train1, test1, 1, save_to, p2, p1)

    train, test = pca_train, pca_test

    # 训练和预测
    # train_and_prediction(models, train1, train1Y, test1, test1Y, names, original, original_ig_train1, original_ig_test1,
    #                      orig_ig, p2,
    #                      p1, new, e_f1, e_f2, new_fs, y2, y1, supplement, x2, x1, supplement_ig, st_x2, st_x1,
    #                      stable_ig)
    train_for_one(models, names, train, train1Y, test, test1Y, pca_with_ori)

    print("################################################################################")

    # 获取各种特征集
    original_ig_train2, original_ig_test2 = get_ori_ig(train2, test2, train2Y, save_to)
    p2, p1 = get_newly_feature(original_ig_train2, original_ig_test2, 2, save_to)
    # e_f1, e_f2, f1, f2 = get_select_newly(p2, p1, train2Y, save_to)
    # x2, x1, y2, y1 = get_ensemble_feature(train2, f1, test2, f2)
    # st_x2, st_x1 = get_stable_feature(original_ig_train2, original_ig_test2, save_to)
    pca_train, pca_test = oripca_ori_new(train2, test2, 2, save_to, p2, p1)
    # single_train, single_test = single_normalizetion(train2, test2)
    # train_3rd, test_2nd = oripca_new_3nd(train2, train2Y, test2, 2, save_to)
    # train = np.hstack([pca_train, single_train, train_3rd])
    # test = np.hstack([pca_test, single_test, test_2nd])
    # train_unio, test_unio = _12_plus_30(train2, test2, 2, save_to, p2, p1)

    train, test = pca_train, pca_test

    # 训练和预测

    # train_and_prediction(models, train2, train2Y, test2, test2Y, names, original, original_ig_train2, original_ig_test2,
    #                      orig_ig, p2,
    #                      p1, new, e_f1, e_f2, new_fs, y2, y1, supplement, x2, x1, supplement_ig, st_x2, st_x1,
    #                      stable_ig)
    train_for_one(models, names, train, train2Y, test, test2Y, pca_with_ori)

    print("################################################################################")

    # 获取各种特征集
    original_ig_train5, original_ig_test5 = get_ori_ig(train5, test5, train5Y, save_to)
    p2, p1 = get_newly_feature(original_ig_train5, original_ig_test5, 5, save_to)
    # e_f1, e_f2, f1, f2 = get_select_newly(p2, p1, train5Y, save_to)
    # x2, x1, y2, y1 = get_ensemble_feature(train5, f1, test5, f2)
    # st_x2, st_x1 = get_stable_feature(original_ig_train5, original_ig_test5, save_to)
    pca_train, pca_test = oripca_ori_new(train5, test5, 5, save_to, p2, p1)
    # single_train, single_test = single_normalizetion(train5, test5)
    # train_3rd, test_2nd = oripca_new_3nd(train5, train5Y, test5, 5, save_to)
    # train = np.hstack([pca_train, single_train, train_3rd])
    # test = np.hstack([pca_test, single_test, test_2nd])
    # train_unio, test_unio = _12_plus_30(train5, test5, 5, save_to, p2, p1)

    train, test = pca_train, pca_test

    # 训练和预测
    # train_and_prediction(models, train5, train5Y, test5, test5Y, names, original, original_ig_train5, original_ig_test5,
    #                      orig_ig, p2,
    #                      p1, new, e_f1, e_f2, new_fs, y2, y1, supplement, x2, x1, supplement_ig, st_x2, st_x1,
    #                      stable_ig)
    train_for_one(models, names, train, train5Y, test, test5Y, pca_with_ori)

    print("################################################################################")

    # 获取各种特征集
    original_ig_train4, original_ig_test4 = get_ori_ig(train4, test4, train4Y, save_to)
    p2, p1 = get_newly_feature(original_ig_train4, original_ig_test4, 4, save_to)
    # e_f1, e_f2, f1, f2 = get_select_newly(p2, p1, train4Y, save_to)
    # x2, x1, y2, y1 = get_ensemble_feature(train4, f1, test4, f2)
    # st_x2, st_x1 = get_stable_feature(original_ig_train4, original_ig_test4, save_to)
    pca_train, pca_test = oripca_ori_new(train4, test4, 4, save_to, p2, p1)
    # single_train, single_test = single_normalizetion(train4, test4)
    # train_3rd, test_2nd = oripca_new_3nd(train4, train4Y, test4, 4, save_to)
    # train = np.hstack([pca_train, single_train, train_3rd])
    # test = np.hstack([pca_test, single_test, test_2nd])
    # train_unio, test_unio = _12_plus_30(train4, test4, 4, save_to, p2, p1)

    train, test = pca_train, pca_test

    # 训练和预测

    # train_and_prediction(models, train4, train4Y, test4, test4Y, names, original, original_ig_train4, original_ig_test4,
    #                      orig_ig, p2,
    #                      p1, new, e_f1, e_f2, new_fs, y2, y1, supplement, x2, x1, supplement_ig, st_x2, st_x1,
    #                      stable_ig)
    train_for_one(models, names, train, train4Y, test, test4Y, pca_with_ori)

    print("################################################################################")

    # 获取各种特征集
    original_ig_train3, original_ig_test3 = get_ori_ig(train3, test3, train3Y, save_to)
    p2, p1 = get_newly_feature(original_ig_train3, original_ig_test3, 3, save_to)
    # e_f1, e_f2, f1, f2 = get_select_newly(p2, p1, train3Y, save_to)
    # x2, x1, y2, y1 = get_ensemble_feature(train3, f1, test3, f2)
    # st_x2, st_x1 = get_stable_feature(original_ig_train3, original_ig_test3, save_to)
    pca_train, pca_test = oripca_ori_new(train3, test3, 3, save_to, p2, p1)
    # single_train, single_test = single_normalizetion(train3, test3)
    # train_3rd, test_2nd = oripca_new_3nd(train3, train3Y, test3, 3, save_to)
    # train = np.hstack([pca_train, single_train, train_3rd])
    # test = np.hstack([pca_test, single_test, test_2nd])
    # train_unio, test_unio = _12_plus_30(train3, test3, 3, save_to, p2, p1)

    train, test = pca_train, pca_test

    # 训练和预测
    # train_and_prediction(models, train3, train3Y, test3, test3Y, names, original, original_ig_train3, original_ig_test3,
    #                      orig_ig, p2,
    #                      p1, new, e_f1, e_f2, new_fs, y2, y1, supplement, x2, x1, supplement_ig, st_x2, st_x1,
    #                      stable_ig)
    train_for_one(models, names, train, train3Y, test, test3Y, pca_with_ori)

    print("################################################################################")
    print("............... Average of results after 5 fold CV in the same order as above .......................")

    for i in range(0, len(models)):
        # print(names[i])
        # print((original[names[i]] / 5) * 100)
        # print((orig_ig[names[i]] / 5) * 100)
        # print((new[names[i]] / 5) * 100)
        # print((new_fs[names[i]] / 5) * 100)
        # print((supplement[names[i]] / 5) * 100)
        # print((supplement_ig[names[i]] / 5) * 100)
        # print((stable_ig[names[i]] / 5) * 100)
        print((pca_with_ori[names[i]] / 5) * 100)
        print("--------------------------")

    print("DONE !!!")
    print(dataset_path)


dataset_path = 'datasets/Ionosphere/ionosphere.csv'
save_to = 'datasets/sonar/'

if __name__ == '__main__':
    models = [KNeighborsClassifier(), LogisticRegression(), svm.LinearSVC(), SVC(kernel='rbf', gamma="auto"),
              RandomForestClassifier(), AdaBoostClassifier(), MLPClassifier(), tree.DecisionTreeClassifier()]
    df = pd.read_csv(dataset_path, header=None)
    df = shuffle(df)
    data = df.sample(frac=1)
    dataset1 = ori_pca(df)
    dataset2 = newly_feature(df)
    dataset3 = A_plus_B(dataset1, dataset2)
    for clf in models:
        acc = single_train(clf, dataset2)
        print(acc*100)
