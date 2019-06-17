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
from models.models import single_train, cross_val_train
from transforms.TheFirstRelationship import single_sin, single_cos, single_tan, ori_sin, ori_cos, ori_tan, \
    single_arcsin, single_arccos, single_arctan, ori_arcsin, ori_arccos, ori_arctan, \
    single_square, ori_square, single_normalizetion, ori_normalizetion
from transforms.TheSecondRelationship import single_pca, ori_pca, newly_feature, ori_newly
from transforms.TheThirdRelationship import get_newly_feature_x_y, ori_newly_feature_x_y
from transforms.unio import A_plus_B
from RL.rl_brain import QLearningTable
from Selection import selectByIG
import copy

np.random.seed(7)


def execu_action(action, dataset, clf, state, actions):
    # state_ = state
    state_ = copy.deepcopy(state)
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
        print("cuttent_acc:", acc)
        print("max_acc:", max_acc)
        print("max_state:", max_state)
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
            q_table.learn(str(state), action, reward, str(state_))
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

dataset_path = 'datasets/sonar/sonar.csv'
save_to = 'datasets/ecoli/'
MAX_EPISODE = 10000  # 训练次数
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
