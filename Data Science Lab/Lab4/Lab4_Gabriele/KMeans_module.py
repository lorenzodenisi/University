import csv
import random
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []
        self.new_centroid = []
        self.labels = {}
        self.output_labels = []

    def fit_predict(self, dots_m, plot_clusters=False, plot_step=5):
        """Run the K-means clustering on X.
        :param plot_clusters:
        :param plot_step:
        :param dots_m: input data points, array, shape = (N,C).
        :return: labels : array, shape = N.
        """
        # Select k points as the initial centroids
        for i in range(self.n_clusters):
            rand_row = random.randint(0, len(dots_m[:, 0]) - 1)
            dot_array = dots_m[rand_row, :]
            self.centroids.append(dot_array)  # Dot_array is a numpy vector

        for n in range(self.max_iter):
            # Init
            self.labels = {}
            self.output_labels = []
            self.new_centroid = []

            # Form k clusters by assigning all points to the closest centroid
            for i in range(len(dots_m[:, 0])):
                first_cycle = True
                for j in range(len(self.centroids)):
                    if first_cycle:
                        min = (((dots_m[i] - self.centroids[j]) ** 2).sum()) ** 0.5
                        min_index = j
                        first_cycle = False
                    else:
                        eucl_dist = (((dots_m[i] - self.centroids[j]) ** 2).sum()) ** 0.5
                        if eucl_dist < min:
                            min = eucl_dist
                            min_index = j
                self.output_labels.append(min_index)
                if min_index not in self.labels.keys():
                    self.labels[min_index] = dots_m[i]
                else:
                    self.labels[min_index] = np.vstack((self.labels[min_index], dots_m[i]))

            # Compute the centroid of each cluster
            for cluster in self.labels.keys():
                points_in_cluster_m = self.labels[cluster]
                new_centroid = points_in_cluster_m.mean(axis=0)
                self.new_centroid.append(new_centroid)

            # until centroids don't change
            condition = np.isclose(self.new_centroid, self.centroids, atol=1e-8)
            for i in range(self.n_clusters):
                if condition[i][0] and condition[i][1]:
                    return self.labels, self.new_centroid, self.output_labels

            # Print every 5 steps if required
            if plot_clusters and n % plot_step == 0:
                self.plot_data(dots_m, self.new_centroid, self.output_labels)

            # Before restarting I need to update centroids
            self.centroids = self.new_centroid
        # END k-means for
        return self.labels, self.centroids, self.output_labels

    def dump_to_file(self, filename, dots_m):
        """Dump the evaluated labels to a CSV file."""
        with open(filename, 'w') as csvfile:
            output = csv.writer(csvfile, delimiter=',')
            for i in range(len(dots_m[:, 0])):
                output.writerow([str(i), str(self.output_labels[i])])

    def plot_data(self, dots_m, centroid_l, labels):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(dots_m[:, 0], dots_m[:, 1], c=labels)
        ax.scatter([el[0] for el in centroid_l], [el[1] for el in centroid_l], marker="+", c='red')
        plt.show()
