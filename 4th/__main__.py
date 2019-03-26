import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
    kmean_ei = 0.0
    kmean_rt = 0.0
    kmean_aa = 0.0
    for i in range(0, iter):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percentage,
                                                            random_state=np.random.random_integers(1, 100))
        k_begin = time.time()
        kmeans = KMeans(n_clusters=3, random_state=0).fit(x_train, y_train)
        kmeans_pred = kmeans.predict(x_test)
        k_end = time.time() - k_begin
        kmean_ei = kmean_ei + metrics.accuracy_score(y_test, kmeans_pred, normalize=False)
        kmean_rt = kmean_rt + k_end
        kmean_aa = kmean_aa + metrics.accuracy_score(y_test, kmeans_pred)
    kmean_ei = kmean_ei / (iter * 1.0)
    kmean_rt = kmean_rt / (iter * 1.0)
    kmean_aa = kmean_aa / (iter * 1.0)
    return kmean_ei, kmean_rt, kmean_aa


def eva_hierarchical(x, y):
    hier_ei = 0.0
    hier_rt = 0.0
    hier_aa = 0.0
    for i in range(0, iter):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percentage,
                                                            random_state=np.random.random_integers(1, 100))
        k_begin = time.time()
        hier = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average").fit(x_train, y_train)
        hier_pred = hier.fit_predict(x_test)
        k_end = time.time() - k_begin
        hier_ei = hier_ei + metrics.accuracy_score(y_test, hier_pred, normalize=False)
        hier_rt = hier_rt + k_end
        hier_aa = hier_aa + metrics.accuracy_score(y_test, hier_pred)
    hier_ei = hier_ei / (iter * 1.0)
    hier_rt = hier_rt / (iter * 1.0)
    hier_aa = hier_aa / (iter * 1.0)
    return hier_ei, hier_rt, hier_aa


def eva_DBSCAN(x, y):
    dbscan_ei = 0.0
    dbscan_rt = 0.0
    dbscan_aa = 0.0
    for i in range(0, iter):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percentage,
                                                            random_state=np.random.random_integers(1, 100))
        k_begin = time.time()
        dbscan = DBSCAN().fit(x_train)
        dbscan_pred = dbscan.fit_predict(x_test)
        k_end = time.time() - k_begin
        dbscan_ei = dbscan_ei + metrics.accuracy_score(y_test, dbscan_pred, normalize=False)
        dbscan_rt = dbscan_rt + k_end
        dbscan_aa = dbscan_aa + metrics.accuracy_score(y_test, dbscan_pred)
    dbscan_ei = dbscan_ei / (iter * 1.0)
    dbscan_rt = dbscan_rt / (iter * 1.0)
    dbscan_aa = dbscan_aa / (iter * 1.0)
    return dbscan_ei, dbscan_rt, dbscan_aa


def visualize(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percentage,
                                                        random_state=np.random.random_integers(1, 100))
    dbscan = DBSCAN().fit(x_train)
    dbscan_pred = dbscan.fit_predict(x_test)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=dbscan_pred)
    plt.title("DBSCAN")
    plt.show()
    hier = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average").fit(x_train, y_train)
    hier_pred = hier.fit_predict(x_test)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=hier_pred)
    plt.title("hierarchical")
    plt.show()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x_train, y_train)
    kmeans_pred = kmeans.predict(x_test)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=kmeans_pred)
    plt.title("kmeans")
    plt.show()
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    plt.title("y_test")
    plt.show()

if __name__ == "__main__":
    x, y = prepare_data()
    kmean_ei, kmean_rt, kmean_aa = eva_kmeans(x, y)
    hier_ei, hier_rt, hier_aa = eva_hierarchical(x, y)
    dbscan_ei, dbscan_rt, dbscan_aa = eva_DBSCAN(x, y)
    print("total iterate:", iter)
    print("method        ", "# of error instances   ", "run_time/s               ", "accuracy/%")
    print("K_means         ", kmean_ei, "                  ", kmean_rt, "     ", kmean_aa)
    print("hierarchical    ", hier_ei, "                  ", hier_rt, "     ", hier_aa)
    print("DBSCAN          ", dbscan_ei, "                  ", dbscan_rt, "     ", dbscan_aa)
    # visualize(x, y)
