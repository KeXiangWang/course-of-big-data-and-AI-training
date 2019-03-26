# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 22:10:28 2019
本程序实现决策树的后剪枝
数据集为西瓜书表4.2，划分为训练集和验证集
参考资料：周志华《机器学习》
@author: yanji
"""

import Decision_Tree_ID3 as TreeID3
import Decision_Tree_Visual as TreeVisual
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# #
# data = load_iris()
# x = data.data
# m_0 = np.mean(x[:, 0])
# m_1 = np.mean(x[:, 1])
# m_2 = np.mean(x[:, 2])
# m_3 = np.mean(x[:, 3])
#
# print(x.shape)
# for i in x:
#     i[0] = 1 if i[0] > m_0 else 2
#     i[1] = 1 if i[1] >m_1 else 2
#     i[2] = 1 if i[2] >m_2 else 2
#     i[3] = 1 if i[3] >m_3 else 2
# #
# y = data.target
# print(y.shape)
# # 仅使用前两列特征
# # x = x[:, :2]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
# np.savetxt("x_train.csv", x_train, delimiter=',')
# np.savetxt("x_test.csv", x_test, delimiter=',')
# np.savetxt("y_train.csv", y_train, delimiter=',')
# np.savetxt("y_test.csv", y_test, delimiter=',')

# with open('data.csv', 'wb') as f:
#     for item in result:
#         line = ','.join(item) + '\n'
#         f.write(line.encode('utf-8'))
# 读取训练集
# watermelonTra=pd.read_csv('watermelon2Training.csv',encoding='gbk')
# #读取验证集
# watermelonVal=pd.read_csv('watermelon2Validation.csv',encoding='gbk')


watermelonTra = pd.read_csv('x_train_2.csv', encoding='gbk')
# 读取验证集
watermelonVal = pd.read_csv('x_test_2.csv', encoding='gbk')
# #
# 基于训练集创建未剪枝决策树
# treeOriginal = TreeID3.creatDecisionTree(watermelonTra)
# print(valPrecision(treeOriginal, watermelonVal))
# # 可视化未剪枝决策树
# TreeVisual.createTree(treeOriginal, '未剪枝决策树')


# 计算决策树在验证集的精度

def valPrecision(thisTree, valdata):
    classTrue = list(valdata.iloc[:, -1])
    valNum = valdata.shape[0]
    classPred = []
    crtNum = 0  # 初始化预测正确样例数
    for rowNum in range(valNum):
        classSple = TreeID3.classify(thisTree, watermelonVal.iloc[rowNum, :])  # 预测该样例的分类
        classPred.append(classSple)
        if classTrue[rowNum] == classSple:  # 判断预分类测是否正确
            crtNum += 1
    return crtNum / valNum  # 返回分类精度


# 对已建立的决策树进行后剪枝
# 递归调用通过设置剪枝代码位置实现自底向顶或自顶向底进行剪枝
def createPostpruningTree(inputTree, dfdata, valdata):
    firstStr = list(inputTree.keys())[0]  # 获取第一个属性值
    secondDict = inputTree[firstStr]
    typedfdata = TreeID3.typeMajority(dfdata)  # 多数表决发确定剩余训练集的类别
    pruningTree = {firstStr: {}}  # 初始化后剪枝决策树
    contrastTree = {firstStr: {}}  # 对该属性建立不划分决策树
    for key in secondDict.keys():
        contrastTree[firstStr][key] = typedfdata  # 不划分决策树即每个属性取值样例的类别都为多数表决结果
        # 以递归调用方式完善决策树
        if type(secondDict[key]).__name__ == 'dict':
            pruningTree[firstStr][key] = createPostpruningTree(secondDict[key],
                                                               TreeID3.splitDataset(dfdata, firstStr, key),
                                                               TreeID3.splitDataset(valdata, firstStr, key))
        else:
            pruningTree[firstStr][key] = secondDict[key]
    # 针对该属性，计算剪枝后与不剪枝决策树在验证集的预测精度
    precisionContrast = valPrecision(contrastTree, valdata)
    precisionPruning = valPrecision(pruningTree, valdata)
    # 将两种决策树进行比较，如果剪枝后能提高精度，则选择对该属性剪枝
    # 剪枝操作放在递归调用之后，实现自底向顶的剪枝
    if precisionContrast > precisionPruning:
        # print(firstStr)
        # print(typedfdata)
        return typedfdata
    else:
        return pruningTree

# 基于训练集创建未剪枝决策树
treeOriginal = TreeID3.creatDecisionTree(watermelonTra)
print('未剪枝决策树', valPrecision(treeOriginal, watermelonVal))
# 可视化未剪枝决策树
TreeVisual.createTree(treeOriginal, '未剪枝决策树')

# 基于未剪枝决策树、训练集与验证集创建后剪枝决策树
treePostpruning = createPostpruningTree(treeOriginal, watermelonTra, watermelonVal)
print('后剪枝决策树', valPrecision(treePostpruning, watermelonVal))
# 后剪枝决策树可视化
TreeVisual.createTree(treePostpruning, '后剪枝决策树')
