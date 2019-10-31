import numpy as np
import matplotlib.pyplot as plt
import math
import KMeans

"""initializing KMeans objects with given K values"""
gauss_obj = KMeans.KMeans(15)
chameleon_obj = KMeans.KMeans(6)

"""loading datasets"""
gauss = np.loadtxt("2D_gauss_clusters.txt", delimiter=',', skiprows=1)
chameleon = np.loadtxt("chameleon_clusters.txt", delimiter=',', skiprows=1)

"""Centroid iterations, in input: (array, plot_clusters, plot_step)"""
gauss_obj.fit_predict(gauss, True, 5)
#chameleon_obj.fit_predict(chameleon, True, 5)

""""Clusters plotting, in input: (array, point size)"""
gauss_obj.scattering_plotting(gauss, 10)
#chameleon_obj.scattering_plotting(chameleon, 10)

"""Writing to file"""
#gauss.dump_to_file("result.csv")
