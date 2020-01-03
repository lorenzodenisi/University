"""
Each of the 5,000 rows contains the x and y coordinates of a single point.
These points are grouped in the Euclidean space in 15 different globular clusters.

k = 15
"""
import numpy as np
from KMeans_package.KMeans_module import KMeans
from Silhouette.Silhouette_module import Silhouette


def main():
    file_name_1 = "gauss_clusters_2D.txt"
    file_name_2 = "chameleon_clusters.txt"
    output_file_name_1 = "out_gauss_clusters.csv"
    output_file_name_2 = "out_chameleon.csv"
    k = 15
    max_iter = 100
    kmeans = KMeans(k, max_iter)

    # Load Data
    dots_matrix = np.loadtxt(file_name_1, delimiter=',', skiprows=1)
    chameleon_ds = np.loadtxt(file_name_2, delimiter=',', skiprows=1)

    # Work on Gauss_clusters
    labels_d, centroid_l, labels_array = kmeans.fit_predict(dots_matrix, plot_clusters=False)
    kmeans.dump_to_file(output_file_name_1, dots_matrix, labels_array)
    kmeans.plot_data(dots_matrix, centroid_l, labels_array, png=True, of="Gauss_cluster.png")

    # Work on Chameleon Clusters
    # labels_d, centroid_l, labels_array = kmeans.fit_predict(chameleon_ds, plot_clusters=False)
    # kmeans.dump_to_file(output_file_name_2, chameleon_ds, labels_array)
    # kmeans.plot_data(chameleon_ds, centroid_l, labels_array, png=True, of="chameleon.png")

    # Second Exercise Silhouette
    silh = Silhouette(dots_matrix, labels_d, labels_array)
    s = silh.silhouette_samples()
    silh.plot_silhouette()


if __name__ == '__main__':
    main()
