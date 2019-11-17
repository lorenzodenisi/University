from sklearn.tree import DecisionTreeClassifier
import numpy as np


# extraction of N samples WITH replacement from train dataset
def extract_train_data(X, y, N):
    indexes = np.array(np.random.choice(range(len(X)), N))
    return np.array(X)[indexes.astype(int)], np.array(y)[indexes.astype(int)]


class MyRandomForestClassifier:
    def __init__(self, n_estimators, max_features):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = [DecisionTreeClassifier(max_features=max_features) for _ in range(n_estimators)]  # init of trees

    # train the trees of this random forest using subsets of X (and y)
    def fit(self, X, y):
        for tree in self.trees:
            X_train, y_train = extract_train_data(X, y, len(X))
            tree.fit(X_train, y_train)

    # predict the label for each point in X
    def predict(self, X):
        total_preds = []
        for x in X:  # for each image
            preds = [0] * 10
            for tree in self.trees:  # for each tree of the forest
                label = tree.predict((x,))  # predict the label
                preds[int(label)] += 1
            total_preds.append(np.argmax(preds))  # the label is chosen by majority (ties are not considered)

        return total_preds

    def features_importances(self):

        importance = np.zeros(784)  # 28*28
        for tree in self.trees:
            importance += tree.feature_importances_  # summation of the feature importance for every tree of the forest

        return importance / sum(importance)  # normalization
