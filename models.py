from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from Data import splitData2xy


def spliteData(dataset):
    X, y = splitData2xy(dataset)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
    return train_x, test_x, train_y, test_y


def single_train(clf, dataset):
    '''
    单次模型训练验证
    :param clf: 模型
    :param dataset: 数据集
    :return: acc
    '''
    train_x, test_x, train_y, test_y = spliteData(dataset)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    acc = accuracy_score(test_y, y_pred)
    return acc


def cross_val_train(clf, dataset):
    X, y = splitData2xy(dataset)
    score = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    return score
