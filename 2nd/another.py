import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pydotplus
from sklearn.datasets import load_iris
import pdb

# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
iris_feature_E = ['sepal length', 'sepal width', 'petal length', 'petal width']
iris_feature = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

import math


class DecisionTreeClassifier(object):
    def __init__(self, criterion):
        self.criterion = criterion
        #
        self.order = []
        #
        self.degree = 16
        self.de1 = np.zeros([self.degree, x_train.shape[1]])
        self.rate = np.zeros([self.degree, x_train.shape[1]])
        self.feature = []
        self.classes_num = np.zeros([self.degree, self.degree])
        self.max = np.zeros(x_train.shape[1])
        self.min = np.zeros(x_train.shape[1])

    def info(self, y_train):
        Ent_D = 0
        for i in range(3):
            sum_t = np.sum(y_train == i)
            temp = y_train.shape[0]
            if temp ==0:
                pk = 0
            else:
                pk =  sum_t / temp
            # pdb.set_trace()
            if pk > 0:
                Ent_D = Ent_D - pk * math.log(pk, 2)
        return Ent_D

    def best_feature(self, x_train, y_train):
        Ent_D = 0
        # 计算当前所有样本的信息熵
        for i in range(3):
            sum_t = np.sum(y_train == i)
            temp = y_train.shape[0]
            pk =  sum_t/ temp
            Ent_D = Ent_D - pk * math.log(pk, 2)
        gain = []
        for k in range(x_train.shape[1]):
            gain_var = 0
            for i in range(self.degree):
                self.rate[i, k] = np.sum(x_train[:, k] == i)
                # pdb.set_trace()
                ent = self.info(y_train[x_train[:, k] == i])
                gain_var = gain_var + self.rate[i, k] / y_train.shape[0] * ent
            gain.append(Ent_D - gain_var)
        # pdb.set_trace()
        # 划分数据，得到信息熵
        gain_sort = sorted(gain)
        for k in range(x_train.shape[1]):
            #
            self.feature.append(np.where(gain == gain_sort[x_train.shape[1] - 1 - k])[0][0])

        # pdb.set_trace()
        return 0

    def normal(self, x_train, flag=0):
        for k in range(x_train.shape[1]):
            if flag == 0:
                x1_max = max(x_train[:, k])
                x1_min = min(x_train[:, k])

                for j in range(self.degree):
                    self.de1[j, k] = x1_min + (x1_max - x1_min) / self.degree * j
            else:
                x1_max = self.max[k]
                x1_min = self.min[k]

                # pdb.set_trace()
            var = x_train[:, k].copy()

            for j in range(self.degree):
                var[x_train[:, k] >= self.de1[j, k]] = j

            x_train[:, k] = var
            if (flag == 0):
                self.min[k] = x1_min
                self.max[k] = x1_max
        return x_train

    def argmax(self, y_train):
        maxnum = 0
        for i in range(4):
            a = np.where(y_train == i)
            # pdb.set_trace()
            if a[0].shape[0] > maxnum:
                maxnum = i
        return maxnum

    def fit(self, x_train, y_train):
        # np.savetxt("x_train_0.csv", x_train, delimiter=',')
        x_train = self.normal(x_train, flag=0)
        # pdb.set_trace()
        # np.savetxt("x_train_1.csv", x_train, delimiter=',')
        self.best_feature(x_train, y_train)
        for i in range(self.degree):
            a = np.where(x_train[:, self.feature[0]] == i)
            # print(a)
            for j in range(self.degree):
                if a != []:
                    b = []
                    for k in a[0]:
                        # pdb.set_trace()
                        if x_train[k, self.feature[1]] == j:
                            b.append(k)

                    print("new")
                    print(b)

                    # pdb.set_trace()
                    if b != []:
                        self.classes_num[i, j] = self.argmax(y_train[b])
                        # pdb.set_trace()
                    else:
                        # 在没有数据子集的时候选择上层中样本数最多的作为分类的结果
                        self.classes_num[i, j] = self.argmax(y_train[a[0]])
                else:
                    self.classes_num[i, j] = self.argmax(y_train)
        # 计算所有的叶结点
        # pdb.set_trace()
        return 0

    def prune(self, x_train, y_train):
        x_train = self.normal(x_train, flag=0)
        # pdb.set_trace()
        self.best_feature(x_train, y_train)
        for i in range(self.degree):
            a = np.where(x_train[:, self.feature[0]] == i)
            # print(a)
            for j in range(self.degree):
                if a != []:
                    b = []
                    for k in a[0]:
                        # pdb.set_trace()
                        if x_train[k, self.feature[1]] == j:
                            b.append(k)

                    print("new")
                    print(b)

                    # pdb.set_trace()
                    if b != []:
                        self.classes_num[i, j] = self.argmax(y_train[b])
                        # pdb.set_trace()
                    else:
                        # 在没有数据子集的时候选择上层中样本数最多的作为分类的结果
                        self.classes_num[i, j] = self.argmax(y_train[a[0]])
                else:
                    self.classes_num[i, j] = self.argmax(y_train)
        # 计算所有的叶结点
        # pdb.set_trace()
        return 0

    def predict(self, x_test):
        # 属性划分,使用递归的方法
        y_show_hat = np.zeros([x_test.shape[0]])
        x_test = self.normal(x_test, 1)

        for j in range(x_test.shape[0]):
            var = int(x_test[j, self.feature[0]])
            var2 = int(x_test[j, self.feature[1]])
            # pdb.set_trace()
            y_show_hat[j] = self.classes_num[var, var2]
        # pdb.set_trace()
        return y_show_hat


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    data = load_iris()
    x = data.data
    print(x.shape)
    y = data.target
    print(y.shape)
    # 仅使用前两列特征
    x = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)  # 训练数据

    y_test_hat = model.predict(x_test)  # 测试数据

    # pdb.set_trace()
    print('accuracy_score:', accuracy_score(y_test, y_test_hat))

    # 画图
    N, M = 50, 50  # 横纵各采样多少个值
    x1_min, x2_min = [min(x[:, 0]), min(x[:, 1])]
    x1_max, x2_max = [max(x[:, 0]), max(x[:, 1])]

    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    print(x_show.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)  # 预测值
    # print( y_show_hat.shape)
    # print(y_show_hat)
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    print(y_show_hat)
    plt.figure(1, figsize=(10, 4), facecolor='w')
    plt.subplot(1, 2, 1)  # 1行2列的第一张子图
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=cm_dark,
                marker='*')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel('sepal length', fontsize=15)
    plt.ylabel('sepal width', fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)

    # plt.title('class', fontsize=17)
    # plt.show()
    # # 训练集上的预测结果
    # y_test = y_test.reshape(-1)
    # print(y_test_hat)
    # print(y_test)
    # result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
    # acc = np.mean(result)
    # print(u'准确度: %.2f%%' % (100 * acc))
    # # 过拟合：错误率
    # depth = np.arange(1, 15)
    # err_list = []
    # for d in depth:
    #     clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
    #     clf.fit(x_train, y_train)
    #     y_test_hat = clf.predict(x_test)  # 测试数据
    #     result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
    #     err = 1 - np.mean(result)
    #     err_list.append(err)
    #     # print d, ' 准确度: %.2f%%' % (100 * err)
    #     print(d, u' 错误率: %.2f%%' % (100 * err))
    # plt.subplot(1, 2, 2)  # 1行2列的第2张子图
    # plt.plot(depth, err_list, 'ro-', lw=2)
    # plt.xlabel('Depth', fontsize=15)
    # plt.ylabel('Error ratio', fontsize=15)
    # plt.title('decision tree', fontsize=17)
    # plt.grid(True)
    # plt.show()
