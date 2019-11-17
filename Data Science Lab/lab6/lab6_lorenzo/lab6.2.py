import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import csv


def main():
    data = read_data("2d-synthetic.csv")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(data["x0"], data["x1"], c=data["label"], s=5)
    tree = DecisionTreeClassifier(max_features=2)
    X = [(x0, x1) for x0, x1 in zip(data["x0"], data["x1"])]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)  # holdout

    tree.fit(X_train, y_train)  # training
    y_pred = tree.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(generate_dot(tree, features=["x0", "x1"]))

    lefts = tree.tree_.children_left
    rights = tree.tree_.children_right
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold

    # get all leaves
    leaves = [i for i in range(tree.tree_.node_count) if (lefts[i] == -1 and rights[i] == -1)]

    for leaf in leaves:
        # for every leaf I get the boundaries of the two features according to threshold of fathers of the leaf
        x_min, x_max, y_min, y_max = get_thresholds(leaf, lefts, rights, thresholds, features)

        # I draw the rectangle
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False)
        ax.add_patch(rect)
    plt.show()

    # To reduce the dimension of the tree
    # we could compute x0+x1 and train the decision tree on that

    X = [[x0 + x1] for x0, x1 in zip(data["x0"], data["x1"])]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    tree2 = DecisionTreeClassifier()

    # in this way we have 100% accuracy and the tree is much smaller than the previous one
    tree2.fit(X_train, y_train)
    y_pred = tree2.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(generate_dot(tree2, features=["x0+x1"]))


def read_data(path):
    data = {
    }
    headers = []
    with open(path) as f:
        header = True
        for cols in csv.reader(f):
            if header:
                header = False
                headers = cols.copy()
                for h in headers: data[h] = []
                continue
            for n, col in enumerate(cols):
                data[headers[n]].append(float(col))
    return data


def generate_dot(dt, features):
    return export_graphviz(dt, feature_names=features)


def get_thresholds(leaf, lefts, rights, thresholds, features):
    # init boundaries
    x_min = 0
    y_min = 0
    x_max = 10
    y_max = 10

    dinasty = []  # all the fathers of leaf in the tree
    less_more = []  # less than threshold or more than threshold (0/1)

    father = leaf

    while father != 0:  # continue until we reach root
        leaf = father
        for n, node in enumerate(lefts):  # check if I'm a left child
            if node == leaf:
                dinasty.append(n)  # if so, I append my father to the dinasty
                father = n
                less_more.append(0)  # I'm a left child so less_more = 0
                break
        for n, node in enumerate(rights):  # check if I'm a right child
            if node == leaf:
                dinasty.append(n)  # if so, I append my father to the dinasty
                father = n
                less_more.append(1)  # I'm a right child so less_more = 1

    dinasty.reverse()  # inversion of order (I need to start from root)
    less_more.reverse()

    for father, direction in zip(dinasty, less_more):
        # update boundaries according to fathers ( from root to the last node before leaf)
        if direction == 0:
            if features[father] == 0:
                x_max = thresholds[father]
            else:
                y_max = thresholds[father]
        else:
            if features[father] == 0:
                x_min = thresholds[father]
            else:
                y_min = thresholds[father]

    return x_min, x_max, y_min, y_max


if __name__ == "__main__":
    main()
