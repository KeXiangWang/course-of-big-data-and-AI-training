from numpy import *
import operator
from math import log


### preparing the data ###
def createDataSet():
    fr = open(r'F:\ResearchData\MyCode\Python\trees\lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    dataSet = lenses
    # the last column of dataSet denotes the y or the labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    labels = lensesLabels
    # the labels mark the former columns in dataSet (features)
    return dataSet, labels

    ### training ###


# calculate the shannon entropy of dataSet
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # type(dataSet) is list
    # the number of rows, N
    labelCounts = {}  # dict
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

    # choose the best feature to split the dataSet


# split the dataSet according the value of feature of dataSet
# more details see P61
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last one denotes the label

    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # the reason we use -1 is due to that it can be set as a mark to check whether the function progresses well
    # that the output is -1 after calling this function denotes it is wrong
    for ii in range(numFeatures):
        featList = [example[ii] for example in dataSet]  # return the data of the ii-th feature in dataSet as a list
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, ii, value)
            prob = len(subDataSet) / float(len(dataSet))  # len(subDataSet) denotes abs(D_ii)
            newEntropy += prob * calcShannonEnt(subDataSet)  # calcShannonEnt(subDataSet) obtains the H(D_ii)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = ii
    return bestFeature

    # majority voting for each leaf note


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

    # creating new tree


# the input para labels is not necessary, just for well see the meaning
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # for the leaf nodes
    # for the case(1) in Alg5.2 in P63
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # for the case(2) in Alg5.2 in P63
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # for the iterations (3)-(6) in Alg5.2 in P64
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # delect an element, bestFeatLabel, in this iteration
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    labels.insert(bestFeat, bestFeatLabel)  # recover the labels from del(labels[bestFeat])
    return myTree

    ### testing ###


# classify
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                class_label = classify(secondDict[key], featLabels, testVec)
            else:
                class_label = secondDict[key]
    return class_label

    ### using ###


# storing the built trees
def storeTree(inputTree, filename):
    import pickle  # a module
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

    def grabTree(filename):
        import pickle

    fr = open(filename)
    return pickle.load(fr)
