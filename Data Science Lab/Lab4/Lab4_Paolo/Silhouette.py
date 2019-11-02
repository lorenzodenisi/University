from scipy.spatial import distance
import numpy as np
import csv
import math
import time

class Silhouette:
    def __init__(self, labels):
        self.n_clusters = max(labels) +1
        self.cluster_sets = [list() for _ in range(self.n_clusters)]
        pass

    def silhouette_samples(self, X, labels):
        """Evaluate the silhouette for each point and return them as a list.
            :param X: input data points, array, shape = (N,C).
            :param labels: the list of cluster labels, shape = N.
            :return: silhouette : array, shape = N
            """

        assert len(labels) == len(X)  # Check if the label list length matches the X length.

        silhouette_samples_list = list()

        print("Number of clusters: " + str(self.n_clusters))
        print("Min and Max: " + str(min(labels)) + ", " + str(max(labels)))
        n = len(labels)

        matrix = np.zeros((n, n))  # Pre-allocate a data-matrix for computing later euclidean distances.
        print('Computing cartesian product...')

        for ii in range(n):  # Iterate over all samples
            point_ii = X[ii]
            if ii % (n/100) == 0:  # Print calculation percentage
                print(str(ii/n*100) + "% completed")
            for jj in range(ii+1, n):  # Look at just the samples that comes after the current sample considered,
                point_jj = X[jj]
                matrix[ii][jj] = distance.euclidean(point_ii, point_jj)  # More efficient than the one used in KMeans
                matrix[jj][ii] = matrix[ii][jj]  # These distances are the same, so we save processing time

        for pos in range(len(labels)):
            self.cluster_sets[labels[pos]].append(pos)

        print('Calculating si quantity for each data sample...')
        for pos in range(len(labels)):  # Silhouette calculation for every point
            bi_list = list()
            for cls_no, cluster in enumerate(self.cluster_sets):
                if len(cluster) == 0: continue  # If cluster is empty go to next iteration
                if cls_no == labels[pos]:
                    ai = sum(matrix[pos][cluster]) / (len(cluster) - 1)
                else:
                    bi = np.mean(matrix[pos][cluster])
                    bi_list.append(bi)
            bi = min(bi_list)
            si = (bi - ai) / max([ai, bi])

            silhouette_samples_list.append(si)
            print("Data sample no.:" + str(pos) + "cls no.: " + str(labels[pos]) + " ai: " + str(ai) + " bi: " + str(bi) + " si: " + str(si))

        return silhouette_samples_list

    def silhouette_score(self, X, labels):
        """Evaluate the silhouette for each point and return the mean.
        :param X: input data points, array, shape = (N,C).
        :param labels: the list of cluster labels, shape = N.
        :return: silhouette : float
        """
        silhouette_samples_list = self.silhouette_samples(X, labels)
        score_over_all_data = np.mean(np.array(silhouette_samples_list))
        
        return score_over_all_data
