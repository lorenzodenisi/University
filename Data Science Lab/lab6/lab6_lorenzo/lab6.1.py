from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def main():
    X, y, features = load_data()
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    dot_code = generate_dot(dt, features)
    print(dot_code)
    y_pred = dt.predict(X)
    accuracy = accuracy_score(y_pred, y)
    print("Accuracy: " + str(accuracy))
    # it's one because we test the model on the training set, if the model is overfitted, the result will be this one

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # training
    dt2 = DecisionTreeClassifier()
    dt2.fit(X_train, y_train)
    y_pred = dt2.predict(X_test)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy:" + str(accuracy))
    print("Recall:" + str(recall))
    print("Precision:" + str(precision))
    print("F1:" + str(f1))

    print(classification_report(y_pred, y_test))

    # Grid Search

    params = {
        "max_depth": [None, 2, 4, 6],
        "splitter": ["best", "random"]
    }

    best_tree = evaluate_gridsearch(params, X_train, X_test, y_train, y_test)

    dot_code = generate_dot(best_tree, features)
    print(dot_code)

    i = extract_feature_importance(best_tree)
    print("Importance: \n" + str(i))

    # the results coincides with the value precomputed inside the tree (leaf nodes are not considered in that case)
    # print(best_tree.feature_importances_)


def load_data():
    dataset = load_wine()
    X = dataset["data"]
    y = dataset["target"]
    feature_names = dataset["feature_names"]

    return X, y, feature_names


def generate_dot(dt, features):
    return export_graphviz(dt, feature_names=features)


def evaluate_gridsearch(params, X_train_valid, X_test, y_train_valid, y_test):
    configs = ParameterGrid(params)  # generate all the combinations

    kf = KFold(5)  # partitioner for the subsets
    results = []

    for train_indices, validation_indices in kf.split(X_train_valid):
        # each cycle it partitions the set into 4 train and 1 validation subsets (each time a different shuffle)
        X_train = X_train_valid[train_indices]
        X_valid = X_train_valid[validation_indices]
        y_train = y_train_valid[train_indices]
        y_valid = y_train_valid[validation_indices]

        fold_results = []

        for config in configs:
            clf = DecisionTreeClassifier(**config)  #hyperparameters, the parameters inside the dictionary must be named as the function parameters
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_valid)
            accuracy = accuracy_score(y_valid, y_pred)
            fold_results.append((accuracy, config))

        results.append(fold_results)

    kfold_res = []  # all results with different configs and different validation sets
    for i in range(len(results[0])):  # for every config
        config = ([], [])
        for j in range(len(results)):  # for every validation set
            config[0].append(results[j][i][0])  # append accuracy score
        config[1].append(results[j][i][1])  # add the config info
        kfold_res.append(config)  # add tuple to the list of all results

    kfold_res = [(np.mean(x[0]), x[1]) for x in kfold_res]  # substitute the list of accuracy score with the average of them

    kfold_res.sort(key=lambda s: -s[0])  # descending sorting
    print("Best overall configuration: " + str(kfold_res[0]))

    clf = DecisionTreeClassifier(**kfold_res[0][1][0])  # now evaluate the best config with test data
    clf.fit(X_train_valid, y_train_valid)  # train it with the usual method
    y_pred = clf.predict(X_test)  # predict test data
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: " + str(accuracy))
    return clf


def extract_feature_importance(tree):
    lefts = tree.tree_.children_left  # lefts[i] specifies the index of the left children of the i-th node
    rights = tree.tree_.children_right  # right[i] specifies the index of the right children of the i-th node
    features = tree.tree_.feature  # features[i] indicates the feature used for the i-th node
    impurities = tree.tree_.impurity  # same concept for the others arrays
    cardinality = tree.tree_.n_node_samples

    N = cardinality[0]
    importance = [0] * tree.tree_.capacity
    for i in range(len(importance)):
        i_p = impurities[i]
        i_l = impurities[lefts[i]] if lefts[i] != -1 else 0
        i_r = impurities[rights[i]] if rights[i] != -1 else 0

        P = cardinality[i]
        L = cardinality[lefts[i]] if lefts[i] != -1 else 0
        R = cardinality[rights[i]] if rights[i] != -1 else 0

        importance[features[i]] += P * i_p / N - L * i_l / N - R * i_r / N
    return importance / sum(importance)


if __name__ == "__main__":
    main()
