import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score
from myRandomForestClassifier import MyRandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time


def main():
    data = load_data()
    # example = (np.asarray(data["X"][0])).reshape(28, 28)
    # plt.imshow(example, cmap="gray", vmin=0, vmax=255)
    # plt.show()

    # holdout
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], train_size=60000)

    # single tree evaluation
    eval_single_dt(X_train, X_test, y_train, y_test)

    # Evaluation of random forest classifiers
    # Firstly my implementation
    tic = time.time()
    evaluate_my_rf(X_train, X_test, y_train, y_test)
    print("Elapsed:" + str(time.time() - tic))

    # Then the sklearn implementation (MUCH faster!!!)
    tic = time.time()
    evaluate_rf(X_train, X_test, y_train, y_test)
    print("Elapsed:" + str(time.time() - tic))

    # Features importance visualization
    rf = MyRandomForestClassifier(10, 28)
    rf.fit(X_train, y_train)
    features_importances = rf.features_importances()
    sns.heatmap(np.reshape(features_importances, (28, 28)), cmap='YlGnBu_r')    # Fancy cmap :)
    plt.show()
    return


def load_data():
    data = {
        "X": [],
        "y": []
    }
    paths = ["mnist_test.csv", "mnist_train.csv"]
    for path in paths:
        with open(path) as f:
            header = True
            for cols in csv.reader(f):
                if header:
                    header = False
                    continue
                data["X"].append(np.array((cols[1:]), dtype="uint8"))
                data["y"].append(int(cols[0]))

    return data


def eval_single_dt(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print(classification_report(y_test, y_pred))
    # print(export_graphviz(dt))


def generate_dot(dt, features):
    return export_graphviz(dt, feature_names=features)


def evaluate_my_rf(X_train, X_test, y_train, y_test):
    # evaluate with myRandomForestClassifier
    precision = []
    accuracy = []
    f1 = []
    recall = []

    for n_trees in range(10, 110, 10):
        print("Classification with " + str(n_trees) + " trees")
        random_forest = MyRandomForestClassifier(n_trees, 28)
        random_forest.fit(X_train, y_train)

        y_pred = random_forest.predict(X_test)

        print(classification_report(y_test, y_pred))
        precision.append(precision_score(y_pred, y_test, average="macro"))
        accuracy.append(accuracy_score(y_pred, y_test))
        f1.append(f1_score(y_test, y_pred, average="macro"))
        recall.append(recall_score(y_test, y_pred, average="macro"))

    plt.plot(range(10, 110, 10), precision, label="precision")
    plt.plot(range(10, 110, 10), recall, label="recall")
    plt.plot(range(10, 110, 10), accuracy, label="accuracy")
    plt.plot(range(10, 110, 10), f1, label="f1")
    plt.legend(loc="best")
    plt.show()

    return precision, recall, accuracy, f1


def evaluate_rf(X_train, X_test, y_train, y_test):
    # evaluate with sklearn RandomForestClassifier
    precision = []
    accuracy = []
    f1 = []
    recall = []
    for n_trees in range(10, 110, 10):
        print("Classification with " + str(n_trees) + " trees")
        random_forest = RandomForestClassifier(n_trees, max_features=28)
        random_forest.fit(X_train, y_train)

        y_pred = random_forest.predict(X_test)

        print(classification_report(y_test, y_pred))
        precision.append(precision_score(y_pred, y_test, average="macro"))
        accuracy.append(accuracy_score(y_pred, y_test))
        f1.append(f1_score(y_test, y_pred, average="macro"))
        recall.append(recall_score(y_test, y_pred, average="macro"))

    plt.plot(range(10, 110, 10), precision, label="precision")
    plt.plot(range(10, 110, 10), recall, label="recall")
    plt.plot(range(10, 110, 10), accuracy, label="accuracy")
    plt.plot(range(10, 110, 10), f1, label="f1")
    plt.legend(loc="best")
    plt.show()

    return precision, recall, accuracy, f1


if __name__ == "__main__":
    main()
