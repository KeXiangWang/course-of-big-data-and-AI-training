import numpy as np
import pickle


def regressEvaluation(tree, test):
    return float(tree)


# =============================================================================
# 数据加载
# =============================================================================

def regressData(filename):
    fr = open(filename)
    # 持久化
    return pickle.load(fr)

# 划分后左右数据集的总误差平方和
def regressErr(data):
    # 回归树的叶子节点取的是均值
    # 计算误差相当于（每个输出－均值）^2
    # 就是方差
    # np.var的方差分母是(n-1)
    return np.var(data[:, -1]) * (np.shape(data)[0] - 1)


# 模型树误差
def modelErr(data):
    w, x, y = linearSolve(data)
    y_pie = x * w
    return sum(np.power(y - y_pie, 2))

# 输出叶结点平均值
def regressLeaf(data):
    # 数据集输出列即最后一列的平均值
    return np.mean(data[:, -1])


# 输出系数
def modelLeaf(data):
    w, x, y = linearSolve(data)
    return w



# =============================================================================
# 创建CART树(回归树或模型树)
# 输入：数据集data，叶子节点形式leafType：regressLeaf（回归树）、modelLeaf（模型树）
# 损失函数errType:误差平方和也分为regressLeaf和modelLeaf
# 用户自定义阈值参数：误差减少的阈值,子样本集应包含的最少样本个数
# =============================================================================
def CreateCart(data, leafType=regressLeaf, errType=regressErr, threshold=(1, 4)):
    # 寻找最优特征与最优切分点
    feature, value = ChooseBest(data, leafType, errType, threshold)
    # 停止条件：当结点的样本个数小于阈值，或基尼指数小于阈值,或者没有更多特征时停止
    if feature == None:
        return value
    returnTree = {}
    returnTree['bestSplitFeature'] = feature
    returnTree['bestSplitValue'] = value
    leftSet, rightSet = binarySplit(data, feature, value)
    returnTree['left'] = leftSet
    returnTree['right'] = rightSet
    return returnTree


def ChooseBest(data, leafType, errType, threshold):
    thresholdErr, thresholdSamples = threshold[0], threshold[1]
    # 数据中输出相同，不需要继续划分树，输出平均值
    # 需要将np array转换成list再做set
    if len(set(data[:, -1].T.tolist()[0])) == 1:
        # 回归树返回叶子平均值，模型树返回系数
        return None, leafType(data)
    m, n = data.shape()
    # 分别处理回归树、模型树的err计算方法
    Err = errType(data)
    bestErr, bestFindex, bestFval = np.Inf, 0, 0
    # 对于每个特征，根据每个取值将data划分成两个子集，计算两个子集的err
    # 取所有可能划分点中，err最小的特征和结点，作为划分点
    # 同时，保证划分的子集中样本个数大于阈值thresholdSamples
    # 这种划分方式适用于连续型数据，标称型数据如果不能比大小则要用id3的方式划分
    for findex in range(n - 1):
        for fval in data[:, findex]:
            left, right = binarySplit(data, findex, fval)
            # 阈值判断
            if (left.shape()[0] < thresholdSamples) or (right.shape()[0] < thresholdSamples):
                continue
            temerr = errType(left) + errType(right)
            # 更新最小误差
            if temerr < bestErr:
                bestErr, bestFindex, bestFval = temerr, findex, fval
    # 检验所选最优划分点的误差，与未划分时的差值是否小于阈值thresholdErr
    if Err - bestErr < thresholdErr:
        return None, leafType(data)

    return bestFindex, bestFval


def binarySplit(data, findex, fval):
    # np.nonzero选取符合条件的非零值所在的每一纬度的下标
    left = data[np.nonzero(data[:, findex] <= fval)[0], :]
    right = data[np.nonzero(data[:, findex] > fval)[0], :]
    return left, right


# =============================================================================
# 回归树、模型树对应的不同处理函数
# =============================================================================

# y=kx+b写成矩阵形式
def linearSolve(data):
    m, n = np.shape(data)
    # mat转换成矩阵，方便后面计算
    x, y = np.mat(np.ones((m, n))), np.mat(np.ones((m, 1)))
    x[:, 1:n] = data[:, 0:(n - 1)]
    y = data[:, -1]
    xTx = x.T * x
    # 计算行列式,判断是否能够取逆
    if np.linalg.det(xTx) == 0:
        # 抛出异常
        raise NameError('matrix cannot do inverse,try increasing the second value of threshold')
    else:
        w = xTx.I * (x.T * y)
        return w, x, y



# =============================================================================
# 剪枝
# =============================================================================
# 从叶子向上，比较剪掉和不剪的误差
def prune(tree, test):
    # 数据集没有数据
    if test.shape()[0] == 0: return getMean(tree)
    # 向下递归到叶子
    if isTree(tree['left']) or isTree(tree['right']):
        testleft, testright = binarySplit(test, tree['bestSplitFeature'], tree['bestSplitFeatValue'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], testleft)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], testright)
    # 找到叶子之后计算误差
    if not isTree(tree['left']) and not isTree(tree['right']):
        leftmean, rightmean = binarySplit(test, tree['bestSplitFeature'], tree['bestSplitFeatValue'])
        errno = sum(np.power(leftmean[:, -1] - tree['left'], 2)) + sum(np.power(rightmean[:, -1] - tree['right'], 2))
        errMerge = sum(np.power(test[:, -1] - getMean(tree), 2))
        if errMerge < errno:
            print
            'merge'
            return getMean(tree)
        else:
            return tree
    else:
        return tree


def getMean(tree):
    # tree['left']保存左子树数据集，如果是叶子节点保存平均值
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.


# 判断是否存在叶结点
def isTree(obj):
    # type(obj)返回的是obj的类型关键字
    # __name__将类型关键字转化为str
    return (type(obj).__name__ == 'dict')


# =============================================================================
# 预测
# =============================================================================
def createForeCast(tree, test, modelEval=regressEvaluation):
    m = len(test)
    y = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y = treeForeCast(tree, test[i], modelEval)
    return y


def treeForeCast(tree, test, modelEval=regressEvaluation):
    # 到了叶子节点输出结果
    if not isTree(tree): return modelEval(tree, test)
    # 向左子树递归
    if test[tree['bestSplitFeature']] <= tree['bestSplitFeature']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], test, modelEval)
        else:
            return modelEval(tree['left'], test)
    # 向右子树递归
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], test, modelEval)
        else:
            return modelEval(tree['right'], test)



if __name__ == '__main__':
    trainfilename = 'e:\\python\\ml\\trainDataset.txt'
    testfilename = 'e:\\python\\ml\\testDataset.txt'

    trainDataset = regressData(trainfilename)
    testDataset = regressData(testfilename)

    cartTree = CreateCart(trainDataset, threshold=(1, 4))
    pruneTree = prune(cartTree, testDataset)
    y = createForeCast(cartTree, np.mat([0.3]), modelEval=regressEvaluation)