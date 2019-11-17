from sklearn.datasets import load_wine
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import statistics
import sys
import time

# 1.1: Loading dataset
dataset = load_wine()

# 1.2 Default classifier
X = dataset["data"]
y = dataset["target"]
feature_names = dataset["feature_names"]
obj = tree.DecisionTreeClassifier()
obj.fit(X, y)


# 1.3: Write tree to file in order to visualize it
a = tree.export_graphviz(obj, feature_names= feature_names)
with open("tree.txt", "w+") as f:
    f.write(a)

# 1.4: accuracy score of classifier
predicted = obj.predict(X)
acc_score = metrics.accuracy_score(y, predicted)

# 1.5: splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# 1.6 training a new model on splitte ddataset
obj2 = tree.DecisionTreeClassifier()
obj2.fit(X_train, y_train)
predicted_with_training = obj2.predict(X_test)
acc_score_predicted = metrics.accuracy_score(y_test, predicted_with_training)
#print(round(acc_score_predicted, 2))
#print(metrics.classification_report(y_test, predicted_with_training))


# 1.7: simple grid search
params = {
    "max_depth": [None, 2, 4, 8],
    "splitter": ["best", "random"]
}

for config in ParameterGrid(params):
    clf = tree.DecisionTreeClassifier(**config)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    #print(config, ":\n", round(accuracy, 2))


# 1.8: cross-validation

# Split the datasets into two:
    # - X_train_valid: the dataset used for the k-fold cross-validation
    # - X_test: the dataset used for the final testing (this will NOT
    # be seen by the classifier during the training/validation phases)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, train_size=0.8)
    kf = KFold(5)  # 5-fold cross-validation
    # X and y are the arrays to be split

best_trees_labels = [0 for x in range(8)]
max_range = 100
now = time.time()
for i in range(max_range):
    list_of_trees = []
    len_p = len(ParameterGrid(params))
    accuracies = []

    for config in ParameterGrid(params):
        list_of_trees.append(config)
        temp_acc = []
        for train_indices, validation_indices in kf.split(X_train_valid):
            X_train = X_train_valid[train_indices]
            X_valid = X_train_valid[validation_indices]
            y_train = y_train_valid[train_indices]
            y_valid = y_train_valid[validation_indices]
            clf = tree.DecisionTreeClassifier(**config)
            clf.fit(X_train, y_train)
            prediction = clf.predict(X_valid)
            accuracy = metrics.accuracy_score(y_valid, prediction)
            temp_acc.append(accuracy)
        accuracies.append(temp_acc)

    final_acc_vector = [statistics.mean(accuracies[i]) for i in range(8)]
    best_config_index = final_acc_vector.index(max(final_acc_vector))
    best_accuracy_mean = final_acc_vector[best_config_index]
    # sys.stdout.write("\r\r\r" + "best config:\n" + str(list_of_trees[best_config_index]) +"\nwith mean = " + "%.2f" % best_accuracy_mean + "\npercentage: " + "%.2f" % (i/max_range*100))
    sys.stdout.write("\r" + "Training " + str(max_range) + " times: " + "%.2f" % (i/max_range*100) +"%")
    best_trees_labels[best_config_index] += 1

best_index_over_n_trials = best_trees_labels.index(max(best_trees_labels))
end = time.time()
print("\nTime elapsed: ", "%.2f" % float(end - now), "s")
print("\nBest tree on ", max_range, " misuratons:\n", list_of_trees[best_index_over_n_trials])


clf = tree.DecisionTreeClassifier(**list_of_trees[best_index_over_n_trials])

for train_indices, validation_indices in kf.split(X_train_valid):
    X_train = X_train_valid[train_indices]
    X_valid = X_train_valid[validation_indices]
    y_train = y_train_valid[train_indices]
    y_valid = y_train_valid[validation_indices]
clf.fit(X_train_valid, y_train_valid)

prediction = clf.predict(X_test)
final_accuracy = metrics.accuracy_score(y_test, prediction)

print(final_accuracy)

#a = tree.export_graphviz(clf, feature_names=feature_names)
