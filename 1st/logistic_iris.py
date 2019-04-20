from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升算法
def gradAscent(dataMat, labelMat):
    m, n = shape(dataMat)
    alpha = 0.1
    maxCycles = 500
    weights = array(ones((n, 1)))

    for k in range(maxCycles):
        a = dot(dataMat, weights)
        h = sigmoid(a)
        error = (labelMat - h)
        weights = weights + alpha * dot(dataMat.transpose(), error)
    return weights


# 随机梯度上升
def randomgradAscent(dataMat, label, numIter=50):
    m, n = shape(dataMat)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 40 / (1.0 + j + i) + 0.2
            randIndex_Index = int(random.uniform(0, len(dataIndex)))
            randIndex = dataIndex[randIndex_Index]
            h = sigmoid(sum(dot(dataMat[randIndex], weights)))
            error = (label[randIndex] - h)
            weights = weights + alpha * error[0, 0] * (dataMat[randIndex].transpose())
            del (dataIndex[randIndex_Index])
    return weights


# 画图
def plotBestFit(weights):
    m = shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
        else:
            xcord2.append(dataMat[i, 1])
            ycord2.append(dataMat[i, 2])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(4, 7, 0.1)
    y = array((-weights[0] - weights[1] * x) / weights[2])
    print(shape(x))
    print(shape(y))
    plt.sca(ax)
    # plt.plot(x, y)  # ramdomgradAscent
    plt.plot(x,y[0])   #gradAscent
    plt.xlabel('SepaLengthCm')
    plt.ylabel('SepalWidthCm')
    plt.title('gradAscent logistic regression')
    # plt.title('ramdom gradAscent logistic regression')
    plt.show()


if __name__ == "__main__":
    # df = pd.read_csv('watermelon_3a.csv')
    # m, n = shape(df)
    # df['idx'] = ones((m, 1))
    #
    # dataMat = array(df[['idx', 'density', 'ratio_sugar']].values[:, :])
    # labelMat = mat(df['label'].values[:]).transpose()
    # weights = gradAscent(dataMat, labelMat)
    # weights = randomgradAscent(dataMat, labelMat)
    data = load_iris()
    dataMat = array(data.data[0:100])
    m,n = shape(dataMat)
    index = ones((m))
    dataMat =  np.insert(dataMat, 0, values=index, axis=1)[:, 0:3]
    labelMat = mat(data.target[0:100]).reshape(-1,1)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
    weights = gradAscent(dataMat, labelMat)
    # weights = randomgradAscent(dataMat, labelMat)
    plotBestFit(weights)
