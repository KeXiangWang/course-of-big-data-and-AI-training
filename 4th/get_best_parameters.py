import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import adjusted_rand_score
import time

np.random.seed()


def prepare_data():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return x, y


percentage = 0.2
iter = 10


def eva_kmeans(x, y):
    parameters = [
        {
            'n_clusters': [3],
            'init': ['k-means++', 'random'],
            'algorithm': ['auto']
        }
    ]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percentage,
                                                        random_state=np.random.random_integers(1, 100))
    kmeans = KMeans()
    clf = GridSearchCV(kmeans, parameters, cv=5, n_jobs=-1)
    clf.fit(x_train, y_train)
    kmeans_pred = clf.predict(x_test)
    print(clf.best_params_)
    print(classification_report(y_test, kmeans_pred, digits=3))


def eva_hierarchical(x, y):
    X, labels_true = x, y
    nums = range(1, 15)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # 链接方式的影响
    linkages = ['ward', 'complete', 'average']
    markers = "+o*"
    for i, linkage in enumerate(linkages):
        ARIs = []
        for num in nums:
            clst = AgglomerativeClustering(n_clusters=num, linkage=linkage)
            # 预测
            predicted_labels = clst.fit_predict(X)
            # ARI指数
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums, ARIs, marker=markers[i], label="linkage:%s" % linkage)

    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()


def eva_DBSCAN(x, y):
    # parameters = [
    #     {
    #         'eps': [0.125, 0.25, 0.5, 1, 2, 4, 8],
    #         'min_samples': range(2, 8),
    #         'metric': ['euclidean', 'mantattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis'],
    #         'algorithm': ['auto']
    #     }
    # ]
    X, labels_true = x, y
    nums = [0.125, 0.25, 0.5, 1,1.5, 2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    metric_ = ['euclidean', 'chebyshev']
    markers = "+o"
    for i, linkage in enumerate(metric_):
        ARIs = []
        for num in nums:
            print(num , "::", linkage)
            clst = DBSCAN(eps=num, metric=linkage)
            predicted_labels = clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums, ARIs, marker=markers[i], label="metric:%s" % linkage)
    ax.set_xlabel("eps")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("DBSCAN")
    plt.show()


if __name__ == "__main__":
    x, y = prepare_data()
    # eva_kmeans(x, y)
    eva_hierarchical(x, y)
    # eva_DBSCAN(x, y)
