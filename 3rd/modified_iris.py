from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time


def mock_data(point_count, point_type=1):
    """
    simulate generating some points
    :param point_count: count of points
    :param point_type: 1:the point can be linearly divided, 2:not
    :return: points, labels
    """
    points = []
    labels = []
    if point_type == 1:
        return [[1, 3], [2, 2.5], [3.5, 1]], [0, 0.1]
    elif point_type == 2:
        for i in range(point_count // 2):
            point_x = random.uniform(0, 10)
            point_y = random.uniform(point_x + 1, 10)
            points.append([point_x, point_y])
            labels.append(0)
        for i in range(point_count // 2):
            point_y = random.uniform(0, 10)
            point_x = random.uniform(point_y + 1, 10)
            points.append([point_x, point_y])
            labels.append(1)
        for i in range(point_count // 6):
            point_y = random.uniform(0, 10)
            point_x = random.uniform(0, 10)
            points.append([point_x, point_y])
            labels.append(random.choice([0, 1]))
    elif point_type == 3:
        for i in range(point_count // 2):
            point_x = random.uniform(-2, 2)
            point_y = random.uniform(-math.sqrt(4 - point_x * point_x), math.sqrt(4 - point_x * point_x))
            points.append([point_x, point_y])
            labels.append(0)
        for i in range(point_count // 2):
            point_x = random.uniform(-2, 2)
            point_y = random.choice([random.uniform(-4, -math.sqrt(4 - point_x * point_x)),
                                     random.uniform(math.sqrt(4 - point_x * point_x), 4)])
            points.append([point_x, point_y])
            labels.append(1)
    else:
        raise Exception("type类型错误{0}".format(point_type))
    return points, labels


def plot_point(dataArr, labelArr, Support_vector_index):
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)

    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='', alpha=0.5, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == "__main__":
    # 读取数据,针对二维线性可分数据
    data = load_iris()
    x = data.data[:, :2]
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
    # dataArr, labelArr = mock_data(100, point_type=2)
    # for i in range(np.shape(dataArr)[0]):
    #     if labelArr[i] == 1:
    #         plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
    #     else:
    #         plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)
    # plt.show()
    # x_train, x_test, y_train, y_test = train_test_split(dataArr, labelArr, test_size=.9, random_state=0)
    # 定义SVM分类器
    for i in range(5):
        c = i * 0.2 + 1
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        # # fit训练数据
        clf.fit(x_train, y_train)
        predict_list = clf.predict(x_test)
        precision = clf.score(x_test, y_test)
        print("Precision is:", precision * 100, "%")
        time.sleep(2)
    # # 获取模型返回值
    # n_Support_vector = clf.n_support_  # 支持向量个数
    # print("支持向量的个数：", n_Support_vector)
    # Support_vector_index = clf.support_  # 支持向量索引
    # # W = clf.coef_  # 方向向量W
    # # b = clf.intercept_  # 截距项b
    # # 绘制分类超平面
    # plot_point(dataArr, labelArr, Support_vector_index)
