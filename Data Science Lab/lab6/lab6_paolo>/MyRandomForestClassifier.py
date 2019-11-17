from sklearn import tree
import numpy as np
import time


class MyRandomForestClassifier:
    def __init__(self, n_estimators, max_features):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees_vector = [tree.DecisionTreeClassifier(max_features=max_features) for i in range(n_estimators)]

    # train the trees of this random forest using subsets of X (and y)
    def fit(self, X, y):
        print("Training forest...")
        now = time.time()
        # Creatig a vector of random labels for each tree of the forest, with replacement parameter = True:
        rand_labels = [np.random.choice(len(X), size=self.max_features, replace=True) for x in range(self.n_estimators)]
        # Creating vectors of training data and labels for each tree based on precomputed random labels:
        y_train_vector = [y[rand_labels[i]] for i in range(self.n_estimators)]
        X_train_vector = [X[rand_labels[i]] for i in range(self.n_estimators)]
        # Training each tree with differents subsets of points
        for x in range(self.n_estimators):
            self.trees_vector[x].fit(X_train_vector[x], y_train_vector[x])
        end = time.time()
        print("%.2f" % float(end - now), "s")

    # predict the label for each point in X
    def predict(self, X):
        print("Forest prediction...")
        now = time.time()
        # Builds a vector of predictions, one for each tree
        forest_prediction = [self.trees_vector[i].predict(X) for i in range(self.n_estimators)]
        pred_by_vote = ['' for i in range(len(X))]
        for x in range(len(X)):  # Selects most common prediction
            # temp_vect = np.zeros(self.n_estimators)
            temp_vect = [0 for i in range(self.n_estimators)]
            for i in range(self.n_estimators):
                temp_vect[int(forest_prediction[i][x])-1] += 1
            pred_by_vote[x] = temp_vect.index(max(temp_vect))
        end = time.time()
        result = np.asarray(pred_by_vote, dtype=str)
        print("%.2f" % float(end - now), "s")
        return result
