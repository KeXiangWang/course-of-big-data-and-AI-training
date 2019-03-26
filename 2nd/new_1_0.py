import xlrd
# import xlwt
import math
import operator
from datetime import date, datetime
from sklearn import datasets


##计算给定数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  # 为所有可能分类创建字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)  # 以2为底数求对数
    return shannonEnt


'''
#创建数据
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
'''


def loaddata():
    f = open("train_feature.txt")
    dataSet = []
    for var in f.readlines():
        dataSet.append(var[:-2].split(' '))
    f.close()
    with open('400words.txt') as f:
        labels = f.read().split(' ')[:-1]
    return dataSet, labels


def createDataSet_iris():
    iris = datasets.load_iris()
    dataSet = []
    for var in iris.data:
        dataSet.append(list(var))
    targets = iris.target
    for index, var in enumerate(targets):
        dataSet[index].append(var)
        labels = ['a', 'b', 'c', 'd']
    return dataSet, labels


def createDataSet():  # 导入数据，存入dataSet放课程分数以及该记录中目标课程是否大于75，大于存‘yes’反之‘no’;features放属性，即所有课程
    data = xlrd.open_workbook("dataset3.xlsx")
    from_sheet = data.sheet_by_index(0)
    id_array = from_sheet.row_values(0)
    print(id_array)
    ncols_length = from_sheet.ncols
    nrows_length = from_sheet.nrows
    del id_array[0]
    del id_array[-1]
    print(id_array)
    features = id_array
    dataSet = [[] for i in range(0, 217)]
    # print dataSet
    i = 0
    for var1 in range(1, from_sheet.nrows):
        temp = from_sheet.row_values(var1)
        for var2 in range(1, from_sheet.ncols):
            if temp[var2] >= '75':
                temp[var2] = '1'
            else:
                temp[var2] = '0'
            dataSet[var1 - 1].append(temp[var2])
    print(dataSet)
    return dataSet, features  ##返回所有数据以及属性


# 依据特征划分数据集  axis代表第几个特征  value代表该特征所对应的值  返回的是划分后的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


'''
#ID3中的做法
#选择最好的数据集(特征)划分方式  返回最佳特征下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):   #遍历特征 第i个
        featureSet = set([example[i] for example in dataSet])   #第i个特征取值集合
        newEntropy= 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #该特征划分所对应的entropy
        infoGain = baseEntropy - newEntropy

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
'''


# 选择最好的数据集(特征)划分方式  返回最佳特征下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainrate = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # 遍历特征 第i个
        featureSet = set([example[i] for example in dataSet])  # 第i个特征取值集合
        newEntropy = 0.0
        splitinfo = 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 该特征划分所对应的entropy
            splitinfo -= prob * math.log(prob, 2)
        if not splitinfo:
            splitinfo = -0.99 * math.log(0.99, 2) - 0.01 * math.log(0.01, 2)
        infoGain = baseEntropy - newEntropy
        infoGainrate = float(infoGain) / float(splitinfo)
        if infoGainrate > bestInfoGainrate:
            bestInfoGainrate = infoGainrate
            bestFeature = i
    return bestFeature


# 创建树的函数代码   python中用字典类型来存储树的结构 返回的结果是myTree-字典
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止继续划分  返回类标签-叶子节点
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 遍历完所有的特征时返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 得到的列表包含所有的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 多数表决的方法决定叶子节点的分类 ----  当所有的特征全部用完时仍属于多类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0;
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 排序函数 operator中的
    return sortedClassCount[0][0]


# 使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # index方法查找当前列表中第一个匹配firstStr变量的元素的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 决策树的存储
def storeTree(inputTree, filename):  # pickle序列化对象，可以在磁盘上保存对象
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):  # 并在需要的时候将其读取出来
    import pickle
    fr = open(filename)
    return pickle.load(fr)
