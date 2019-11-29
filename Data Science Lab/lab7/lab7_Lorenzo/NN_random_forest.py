import spectrogramCNN
import numpy as np


class NNRandomForest:
    n_networks = 0
    networks = []

    def __init__(self, n_networks, input_shape, output_size):
        self.n_networks = n_networks
        self.input_shape = input_shape
        self.output_size = output_size

    def build(self):
        self.networks = [spectrogramCNN.ConvModel(input_shape=self.input_shape, output_size=self.output_size)
                         for _ in range(self.n_networks)]  # init models
        for network in self.networks:
            network.build()  # build the models

    def fit(self, X_train, X_test, y_train, y_test):
        for i, model in enumerate(self.networks):  # for each model
            print(str(i) + " / " + str(len(self.networks)))

            # extraction with replacement for training data
            indexes = np.array(np.random.choice(range(len(X_train)), len(X_train)))
            x_tr = np.array(X_train)[indexes.astype(int)]  # fancy indexing *_*
            y_tr = np.array(y_train)[indexes.astype(int)]

            # fit the models with train data, test data is only used for validation
            model.fit(x_tr, X_test, y_tr, y_test, 100)

    def predict(self, X):
        final_preds = []
        total_preds = []
        for model in self.networks:  # for each network of the forest
            labels = model.predict(X)  # predict the labels
            total_preds.append(labels)

        for i in range(len(total_preds[0])):  # for each record prediction
            count = [0] * self.output_size
            for pred in np.array(total_preds)[:, i]:  # choose the most frequent one
                count[int(pred)] += 1
            final_preds.append(np.argmax(count))
        return final_preds
