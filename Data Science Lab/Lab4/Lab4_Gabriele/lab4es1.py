"""
Each of the 5,000 rows contains the x and y coordinates of a single point.
These points are grouped in the Euclidean space in 15 different globular clusters.

k = 15
"""
import numpy as np
import KMeans_package.KMeans_module as km


def main():
    file_name = "gauss_clusters_2D.txt"
    test_file_name = "test.csv"
    output_file_name = "labels.csv"
    k = 15
    max_iter = 100
    kmeans = km.KMeans(k, max_iter)

    dots_matrix = np.loadtxt(file_name, delimiter=',', skiprows=1)
    # plot_data(dots_matrix)
    labels_d, centroid_l, labels_array = kmeans.fit_predict(dots_matrix, plot_clusters=False)
    print(len(centroid_l))
    # kmeans.dump_to_file(output_file_name, dots_matrix)
    kmeans.plot_data(dots_matrix, centroid_l, labels_array)


if __name__ == '__main__':
    main()
