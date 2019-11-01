import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit_predict(self, X, plot_clusters=False, plot_step=5):
        """Run the K-means clustering on X.
        :param X: input data points, array, shape = (N,C).
        :return: labels : array, shape = N.
        """

        # TODO add check on normalization of input data (-1,1)

        # I select the starting centroid among the already present points, randomly
        centroids_indexes = [random.randint(0, len(X)-1) for _ in range(self.n_clusters)]
        self.centroids = np.array([X[i] for i in centroids_indexes])
        self.labels = [-1] * len(X)

        for count in range(self.max_iter):

            # For every point compute the euclidean distance from the centroids and select the nearest
            for i in range(len(X)):
                dist = (X[i] - self.centroids) ** 2
                dist = np.sum(dist, axis=1)
                dist = np.sqrt(dist)
                self.labels[i] = dist.argmin()

            # For every centroid select all the points labelled to them
            for centroid_index in range(len(self.centroids)):
                points = []
                for n, point in enumerate(X):
                    if self.labels[n] == centroid_index:
                        points.append(point)

                if len(points) > 0:
                    # transposition is used to get columns as rows
                    points_t = (np.array(points)).transpose()
                    # recompute the centroid according to the mean value of x and y of the points
                    self.centroids[centroid_index] = [points_t[0].mean(), points_t[1].mean()]
            if plot_clusters and count % plot_step == 0:
                self.plot(X, self.labels, self.centroids)

            sys.stdout.write('\t\t\t\t\r'+str(count + 1) + "/" + str(self.max_iter))
        print('\n')
        return self.labels, self.centroids

    def dump_to_file(self, filename):
        """Dump the evaluated labels to a CSV file."""

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Id', 'ClusterId'])

            for n, label in enumerate(self.labels):
                writer.writerow([n, label])

    def plot(self, X, labels, centroids):
        X_t = X.transpose()
        plt.rcParams["figure.figsize"] = (7, 7)
        plt.scatter(X_t[0], X_t[1], s=0.5, c=labels)
        C_t = centroids.transpose()
        plt.scatter(C_t[0], C_t[1], marker='*', edgecolors="red")
        plt.figure(figsize=(5, 5))
        plt.show()

    def set_n_clusters(self, n):
        self.n_clusters = n
