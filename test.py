import pandas as pd


def addSomeThings(test, train):
    test.append(0)
    train.append(0)


if __name__ == '__main__':
    # df = pd.read_csv("datasets/Wine/wine.data", header=None)
    # Y = df[df.columns[0]]
    # X = df.drop(df.columns[0], 1)
    # data = pd.concat([X, Y], axis=1)
    # data.to_csv("datasets/Wine/Wine.csv", index=False, header=None)

    # test = pd.read_csv("datasets/Shuttle/shuttle.trn", header=None)
    # train = pd.read_csv("datasets/Shuttle/shuttle.tst", header=None)
    # data = pd.concat([test, train])
    # data.to_csv("datasets/Shuttle/shuttle.csv", index=False, header=None)

    df = pd.read_csv("datasets/sonar/sonar.csv", header=None)
    # print(df.shape)
    a = 2
    b = 3
    for i in range(10):
        b = b + i
        print(i)
