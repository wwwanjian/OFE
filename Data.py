# 数据集处理
import numpy as np

np.random.seed(7)


# 打乱数据集
def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


# 划分数据集为5组
def splitDataToCouples(data):
    n, m = data.shape
    print(n, m)
    x = data.drop(data.columns[len(data.columns) - 1], 1)
    Y = data[data.columns[len(data.columns) - 1]]
    X = x.values
    y = Y.values
    print("Features in Original Dataset:")
    p, pp = X.shape
    # Dividing data into 5 parts where 4 parts are used for training and 1 for testing in each iteration

    train1 = X[:(int)(0.8 * n), :]
    test1 = X[(int)(0.8 * n):, :]

    train2 = X[(int)(0.2 * n):, :]
    test2 = X[:(int)(0.2 * n), :]

    train3 = np.concatenate((X[:(int)(0.6 * n), :], X[(int)(0.8 * n):, :]), axis=0)
    test3 = X[(int)(0.6 * n):(int)(0.8 * n), :]

    train4 = np.concatenate((X[:(int)(0.4 * n), :], X[(int)(0.6 * n):, :]), axis=0)
    test4 = X[(int)(0.4 * n):(int)(0.6 * n), :]

    train5 = np.concatenate((X[:(int)(0.2 * n), :], X[(int)(0.4 * n):, :]), axis=0)
    test5 = X[(int)(0.2 * n):(int)(0.4 * n), :]

    train1Y = y[:(int)(0.8 * n)]
    test1Y = y[(int)(0.8 * n):]

    train2Y = y[(int)(0.2 * n):]
    test2Y = y[:(int)(0.2 * n)]

    list1 = y[:(int)(0.6 * n)]
    list2 = y[(int)(0.8 * n):]
    train3Y = np.append(list1, list2)
    test3Y = y[(int)(0.6 * n):(int)(0.8 * n)]

    list1 = y[:(int)(0.4 * n)]
    list2 = y[(int)(0.6 * n):]
    train4Y = np.append(list1, list2)
    test4Y = y[(int)(0.4 * n):(int)(0.6 * n)]

    list1 = y[:(int)(0.2 * n)]
    list2 = y[(int)(0.4 * n):]
    train5Y = np.append(list1, list2)
    test5Y = y[(int)(0.2 * n):(int)(0.4 * n)]
    return train1, train1Y, test1, test1Y, \
           train2, train2Y, test2, test2Y, \
           train3, train3Y, test3, test3Y, \
           train4, train4Y, test4, test4Y, \
           train5, train5Y, test5, test5Y,
