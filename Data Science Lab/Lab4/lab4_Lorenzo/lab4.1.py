import numpy as np
import matplotlib.pyplot as plt
import kmeans
import time
import sys


def main():
    data = np.loadtxt('chameleon_clusters.txt', delimiter=',', skiprows=1)
    data /= (np.ndarray.max(data) / 2)
    data -= np.ndarray.mean(data)

    # plt.grid(linewidth=0.2)
    # plt.scatter(x, y, s=0.1)
    # plt.savefig('plot.png', dpi=600)
    # plt.show()

    km = kmeans.KMeans(15)
    tic = time.time()
    labels, centroids = km.fit_predict(data)
    print("Elapsed:" + str(time.time() - tic))

    km.dump_to_file('labels.csv')
    # newarray[2] = [mat[2][i] for i in range(len(mat[2]))]
    km.plot(data, labels, centroids)

    # print(silhouette_score(data, labels))
    s = silhouette_samples(data, labels)

    s.sort()
    plt.hist(s, bins=50)
    plt.savefig('silhouette.png')
    plt.show()
    print(str(np.mean(s)))

    # eval_best_cluster_dim(data, range(1, 16))
    pass


def silhouette_samples(X, labels):
    """Evaluate the silhouette for each point and return them as a list.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array, shape = N
    """

    n_clusters = len(np.unique(labels))
    cluster_points = []
    a = [-1] * len(labels)
    b = [-1] * len(labels)
    s = [-1] * len(labels)

    for i in range(n_clusters):
        cluster_points.append([])
        for point, label in zip(X, labels):
            if label == i:
                cluster_points[i].append(point)

    for n, point in enumerate(X):
        cluster = labels[n]

        # evaluating distance from points inside same cluster
        dist = (point - cluster_points[cluster]) ** 2
        dist = dist.sum(axis=1)
        dist = np.sqrt(dist)

        # evaluating a
        a[n] = dist.sum() / (len(dist) - 1)

        outer_dist = [sys.maxsize] * n_clusters
        for j in range(n_clusters):
            if j != cluster:
                # evaluating mean distance from different clusters
                dist = (point - cluster_points[j]) ** 2
                dist = dist.sum(axis=1)
                dist = np.sqrt(dist)
                outer_dist[j] = dist.mean()

        # evaluating b by taking the nearest neighbor cluster
        b[n] = np.min(outer_dist)

        # evaluating point silhouette
        s[n] = (b[n] - a[n]) / max(a[n], b[n])

        # print progress once in a while
        if (n + 1) % 500 == 0:
            sys.stdout.write('\t\t\t\t\r' + str(n + 1) + '/' + str(len(X)))

    return s
    pass


def silhouette_score(X, labels):
    """Evaluate the silhouette for each point and return the mean.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : float
    """
    print('Evaluating silhouette score...')
    return np.mean(silhouette_samples(X, labels))


def eval_best_cluster_dim(X, dims):
    km = kmeans.KMeans(dims[0])
    results = []
    for dim in dims:
        print('\nEvaluating ' + str(dim) + ' clusters')
        km.set_n_clusters(dim)
        labels, _ = km.fit_predict(X, plot_clusters=False)
        results.append(silhouette_score(X, labels))

    plt.plot(dims, results)
    plt.show()


if __name__ == "__main__":
    main()
