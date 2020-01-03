import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import time


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def random_centroid(self, na):
        """Calculates a random centroid between max and min coordinates of the gauss cluster"""
        rc = np.random.uniform(low=np.ndarray.min(na), high=np.ndarray.max(na), size=None)
        return rc

    def euclidean_distance(self, a, b):
        """Euclidean distance calculation"""
        es = (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
        ed = math.sqrt(es)
        return ed

    """Creating cluster plot"""
    def scattering_plotting(self, D, dim):
        # Creating x and y vectors
        x = [D[i][0] for i in range(len(D))]
        y = [D[i][1] for i in range(len(D))]
        # Creating centroid vector
        xc = [self.centroids[i][0] for i in range(len(self.centroids))]
        yc = [self.centroids[i][1] for i in range(len(self.centroids))]
        colors = [self.labels[i] for i in range(len(self.labels))]  # creating colors vector for the K clusters
        plt.clf()  #Clears plot
        plt.scatter(x, y, c=colors, s=dim)  # Plotting points
        plt.scatter(xc, yc, marker="*", color='r')  # Plotting centroids
        plt.show()
        plt.pause(0.001)

    def fit_predict(self, X, plot_clusters, plot_step):
        # Creating random centroids:
        self.centroids = [[self.random_centroid(X), self.random_centroid(X)] for i in range(self.n_clusters)]
        print("Computing new centroids...")
        iterations = 1
        now = time.time()

        """Main iteration loop"""
        for b in range(self.max_iter):
            self.labels = ['' for x in range(len(X))]  # labels is initialized as a list
            for i in range(len(X)):
                #  Building a vector with the euclidean distances between the point i and all centroids
                dist_vector = []
                for j in range(self.n_clusters):

                    dist_vector.append(self.euclidean_distance(self.centroids[j], X[i]))
                #  assignment of the cluster index based on the closest centroid
                self.labels[i] = (np.argmin(dist_vector))

            #  Calculation of the new centroids
            for i in range(self.n_clusters):
                temp_sum = [0, 0]
                mean = [0, 0]  # Default point if cluster has no elements
                n = 0  # Number of points in the cluster i
                for j in range(len(X)):
                    if self.labels[j] == i:
                        # We add together all the points of the cluster i
                        temp_sum[0] += X[j][0]
                        temp_sum[1] += X[j][1]
                        n += 1
                    if n != 0:  # If the cluster is not empty we compute the mean of all its points
                        mean = [(temp_sum[0]/n), (temp_sum[1]/n)]

                self.centroids[i] = mean  # We assign the mean to the corresponding centroid

            print("Iteration " + str(iterations) + "/" + str(self.max_iter))
            iterations += 1  # Counting iteration progress
            if (plot_clusters is True) and (b % plot_step == 0):
                # Step plotting if requested by user
                self.scattering_plotting(X, 10)
        end = time.time()  # Computation time
        print("Time elapsed: ",  int((end - now)/60), "min ", "%.2f" % float((end - now)%60), " sec")

        return np.array(self.labels)

    def dump_to_file(self, filename):
        #Dump the evaluated labels to a CSV file.
        with open(filename, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(['Id', 'ClusterId'])
            for id, a_label in enumerate(self.labels):
                writer.writerow([id, a_label])


