import numpy as np
import matplotlib.pyplot as plt


class Silhouette:

    def __init__(self, X, labels, labels_array):
        self.__dataset = X
        self.__labels_d = labels
        self.__labels_array = labels_array
        self.__s = None

    def silhouette_samples(self):
        """Evaluate the silhouette for each point and return them as a list.
        :param X: input data points, array, shape = (N,C).
        :param labels: the list of cluster labels, shape = N.
        :return: silhouette : array, shape = N
        """
        a = None
        b = None
        s = None

        n_points = len(self.__dataset[:, 0])
        for i in range(n_points):
            cluster_ID = self.__labels_array[i]

            # a array: Evaluating distance from points inside same cluster
            dist = (((self.__dataset[i] - self.__labels_d[cluster_ID]) ** 2).sum(axis=1)) ** 0.5
            cluster_cardinality = len(self.__labels_d[cluster_ID][:, 0])
            inner_dist = dist.sum() / (cluster_cardinality - 1)
            if a is None:
                a = np.array([inner_dist])
            else:
                a = np.hstack((a, inner_dist))

            # b array: Evaluating distance from different clusters
            outer_dist = None
            n_clusters = len(self.__labels_d.keys())
            for j in range(n_clusters):
                dist = (((self.__dataset[i] - self.__labels_d[j]) ** 2).sum(axis=1)) ** 0.5
                if outer_dist is None:
                    outer_dist = np.array([dist.mean()])
                else:
                    outer_dist = np.hstack((outer_dist, dist.mean()))
            b_i = outer_dist.min()
            if b is None:
                b = np.array([b_i])
            else:
                b = np.hstack((b, b_i))

            # evaluating silhouette
            if s is None:
                s = np.array([(b[i] - a[i]) / max(a[i], b[i])])
            else:
                s = np.hstack((s, (b[i] - a[i]) / max(a[i], b[i])))
        self.__s = s
        return s

    def silhouette_score(self, X, labels):
        """Evaluate the silhouette for each point and return the mean.
        :param X: input data points, array, shape = (N,C).
        :param labels: the list of cluster labels, shape = N.
        :return: silhouette : float
        """
        return self.silhouette_samples().mean()

    def plot_silhouette(self, png=False, of="plot_silhouette.png"):
        if self.__s is None:
            s = self.silhouette_samples()
        else:
            s = self.__s
        n_points = len(self.__dataset[:, 0])
        fig, ax = plt.subplots(figsize=(12, 12))
        x_axis = np.linspace(1, n_points, n_points)
        ax.bar(x_axis, np.sort(s))
        plt.show()
        if png:
            fig.savefig(of)

    def get_dataset(self):
        return self.__dataset

    def get_labels(self):
        return self.__labels_d

    def get_labels_array(self):
        return self.__labels_array

    def set_dataset(self, X):
        self.__dataset = X

    def set_labels(self, labels):
        self.__labels_d = labels

    def set_labels_array(self, labels_array):
        self.__labels_array = labels_array
