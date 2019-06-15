# 核心处理文件
import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from data.Data import shuffle, splitDataToCouples
from models.models import single_train,cross_val_train
from transforms.TheFirstRelationship import single_sin, single_cos, single_tan, ori_sin, ori_cos, ori_tan, \
    single_arcsin, single_arccos, single_arctan, ori_arcsin, ori_arccos, ori_arctan, \
    single_square, ori_square, single_normalizetion, ori_normalizetion
from transforms.TheSecondRelationship import single_pca, ori_pca, newly_feature, ori_newly
from transforms.TheThirdRelationship import get_newly_feature_x_y, ori_newly_feature_x_y
from transforms.unio import A_plus_B
from RL.rl_brain import QLearningTable
from Selection import selectByIG

np.random.seed(7)


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


def execu_action(action, dataset, clf, state, actions):
    state_ = state
    state_.append(action)
    acc_ori = single_train(clf, dataset)
    df = actions[action](dataset)
    df = selectByIG(df)
    global max_acc
    global max_state
    try:
        acc = single_train(clf, df)
        if acc > max_acc:
            max_acc = acc
            max_state = state_
        print(acc)
    except Exception as e:
        print(e)
    reward = acc - acc_ori
    return state_, reward, df


def RL_update(q_table, dataset):
    models = [KNeighborsClassifier(), LogisticRegression(), svm.LinearSVC(), SVC(kernel='rbf', gamma="auto"),
              RandomForestClassifier(), AdaBoostClassifier(), MLPClassifier(), tree.DecisionTreeClassifier()]
    model_names = ['kNN', 'LR', 'LSVM', 'PSVM', 'RF', 'AB', 'NN', 'DT']
    n_actions = len(actions)
    clf = models[3]
    for episode in range(MAX_EPISODE):
        df = dataset
        process_flow = []
        state = process_flow
        for i in range(MAX_DEPTH):
            # choose acton
            action = q_table.choose_action(str(state))
            while i < MAX_DEPTH - 1 and action >= 14:
                action = q_table.choose_action(str(state))
            # exec action
            state_, reward, df = execu_action(action, df, clf, state, actions)
            # update qtable
            q_table.learn(str(state_), action, reward, str(state))
            # update state
            state = state_
            print(q_table.q_table)
    print("done!")
    q_table.q_table.to_csv("qtable.csv")
    print("max_acc:", max_acc)
    print("max_state:", max_state)


actions = [single_sin, single_cos, single_tan, ori_sin, ori_cos, ori_tan, single_square, ori_square,
           single_normalizetion, ori_normalizetion, single_pca, ori_pca, get_newly_feature_x_y, ori_newly_feature_x_y,
           newly_feature, ori_newly, ]

dataset_path = 'datasets/Bank/bank.csv'
save_to = 'datasets/ecoli/'
MAX_EPISODE = 10  # 训练次数
MAX_DEPTH = 3  # 深度
max_acc = 0
max_state = []

if __name__ == '__main__':
    # models = [KNeighborsClassifier(), LogisticRegression(), svm.LinearSVC(), SVC(kernel='rbf', gamma="auto"),
    #           RandomForestClassifier(), AdaBoostClassifier(), MLPClassifier(), tree.DecisionTreeClassifier()]
    df = pd.read_csv(dataset_path, header=None)
    df = shuffle(df)
    # # data = df.sample(frac=1)
    # dataset1 = single_arccos(df)
    # dataset2 = newly_feature(df)
    # dataset3 = A_plus_B(dataset1, dataset2)
    # for clf in models:
    #     acc = single_train(clf, dataset1)
    #     print(acc * 100)

    q_table = QLearningTable(actions=list(range(len(actions))))
    RL_update(q_table, df)

    # df = pd.read_csv("qtable.csv",index_col=0)
    # print(df)
