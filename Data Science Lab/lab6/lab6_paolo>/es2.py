import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics

# 2.1: Loading and plotting dataset

dataset_name = "2d-synthetic.csv"
dataset = np.loadtxt(dataset_name, delimiter=',', skiprows=1)
plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2], s = 2)  # Plotting points


# 2.2 plotting data after training on firs n samples

n = 100
clf = tree.DecisionTreeClassifier()
clf.fit(dataset[0:n, 0:2], dataset[0:n, 2])
predicted_with_standard_tree = clf.predict(dataset[:, 0:2])
plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=predicted_with_standard_tree, s = 2)  # Plotting points
plt.show()


