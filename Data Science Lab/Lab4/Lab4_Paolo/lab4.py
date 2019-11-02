import numpy as np
import KMeans
import Silhouette
import matplotlib.pyplot as plt

"""PART 1: K-means design and implementation"""

"""Change file name and K to analyze the other dataset"""
dataset_name = "chameleon_clusters.txt"  # or chameleon_clusters.txt
K = 6  # Or 6

"""initializing KMeans object with given K value"""
dataset_obj = KMeans.KMeans(K)

"""loading dataset."""
dataset = np.loadtxt(dataset_name, delimiter=',', skiprows=1)

"""Centroid iterations, in input: (array, plot_clusters, plot_step)"""
labels_result = dataset_obj.fit_predict(dataset, True, 1)

""""Clusters plotting, in input: (array, point size)"""
dataset_obj.scattering_plotting(dataset, 10)

"""Writing to file"""
dataset_obj.dump_to_file("result.csv")


"""Part 2: Evaluate clustering performance:"""
s = Silhouette.Silhouette(labels_result)  # Initializing silhouette object
silhouette_sample = s.silhouette_samples(dataset, labels_result)  # Creating silhouette list
silhouette_sample.sort()  # Sorting silhouettes list
plt.figure()  #
plt.hist(silhouette_sample, bins=50)  # Plotting silhouette values in ascending order
