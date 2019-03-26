from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import time


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

    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=33)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    # 定义SVM分类器

    # svc = SVC()
    #
    # parameters = [
    #     {
    #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #         'kernel': ['rbf']
    #     },
    #     {
    #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'kernel': ['linear']
    #     },
    #     {
    #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'kernel': ['poly'],
    #         'degree': [1, 2, 3]
    #     }
    # ]
    # # fit训练数据
    # clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)
    for i in [1, 2, 3]:
        c = i
        clf = SVC(C=9.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=c, gamma='auto', kernel='poly',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        clf.fit(x_train, y_train)
        predict_list = clf.predict(x_test)
        precision = clf.score(x_test, y_test)
        print('poly ', 'C:', c, "Precision is:", precision * 100, "%")
    # print(clf.best_params_)
    # time.sleep(2)
    classification_report(y_test, predict_list, target_names=digits.target_names.astype(str))

    # # 获取模型返回值
    # n_Support_vector = clf.n_support_  # 支持向量个数
    # print("支持向量的个数：", n_Support_vector)
    # Support_vector_index = clf.support_  # 支持向量索引
    # # W = clf.coef_  # 方向向量W
    # # b = clf.intercept_  # 截距项b
    # # 绘制分类超平面
    # plot_point(dataArr, labelArr, Support_vector_index)
