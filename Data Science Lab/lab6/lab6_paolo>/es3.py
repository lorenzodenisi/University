from sklearn.datasets import fetch_openml
from sklearn import tree
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from MyRandomForestClassifier import MyRandomForestClassifier
import math
from sklearn import ensemble
import seaborn as sns

# 3.1: load dataset
print("Fetching dataset... ")
now = time.time()
dataset = fetch_openml("mnist_784")
X = dataset["data"]
y = dataset["target"]
end = time.time()
print("%.2f" % float(end - now), "s")

new_X = []
new_y = []
for i in range(len(X)):
    if int(y[i]) == 7 or int(y[i]) == 8:
        new_X.append(X[i])
        new_y.append(y[i])
X = np.asarray(new_X)
y = np.asarray(new_y)


# 3.2: Train simple decision tree

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=6/7)  # Splitting dataset
tree_obj = tree.DecisionTreeClassifier()

print("Training...")
now = time.time()
tree_obj.fit(X_train, y_train) # Training decision tree
end = time.time()
print("%.2f" % float(end - now), "s")

prediction = tree_obj.predict(X_test)  # Prediction
acc_score = metrics.accuracy_score(y_test, prediction)  # Accuracy score
print("Accuracy score: ", "%.2f" % acc_score)

# Dumps graph to txt
tree_graph = tree.export_graphviz(tree_obj)
with open("MNISTtree.txt", "w+") as f:
    f.write(tree_graph)


# 3.3: Random forest
size_of_subsets = round(math.sqrt(len(X_train)))
number_of_trees = 2
math.sqrt(len(X_train))
forest = MyRandomForestClassifier(number_of_trees, size_of_subsets)
forest.fit(X_train, y_train)
forest_prediction = forest.predict(X_test)
forest_accuracy = metrics.accuracy_score(y_test, forest_prediction)  # Accuracy score
print("Accuracy score my forest: ", "%.2f" % forest_accuracy)


# 3.4: Sklearn random forest
print("Training forest...")
now = time.time()
number_of_trees = 100
forest = ensemble.RandomForestClassifier(n_estimators=number_of_trees)
forest.fit(X_train, y_train)
forest_prediction = forest.predict(X_test)
forest_accuracy = metrics.accuracy_score(y_test, forest_prediction)
print("Accuracy score of the forest: ", "%.2f" % forest_accuracy)
end = time.time()
print("%.2f" % float(end - now), "s")

sns.heatmap(np.reshape(forest.feature_importances_, (28, 28)), cmap='binary')