from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
import keras.backend as K


class ConvModel:
    model = None

    def __init__(self, input_shape, output_size):
        # ____________________________________
        # CONFIGURATION FOR GPU-ACCELERATION
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 6})
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        # ___________________________________

        self.input_shape = input_shape
        self.output_size = output_size

    def build(self):
        '''
        the model is made by pipelining convolution, pooling and dropout three times
        (kernel size is not square due to the fact that the spectrogram had dimension (129, 28))
        followed by flattening layer, a dense layer with dropout and a final dense softmax layer for classification

        the used metric is f1 (implemented below)
        '''

        self.model = Sequential()
        self.model.add(
            Conv2D(32, kernel_size=(2, 3), activation='relu', padding='same', strides=1, input_shape=self.input_shape))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(32, kernel_size=(2, 3), activation='relu', padding='same', strides=1))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(32, kernel_size=(2, 3), activation='relu', padding='same', strides=1))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', name='dense'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(self.output_size, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1])
        self.model.summary()

    def fit(self, X_train, X_test, y_train, y_test, epochs):
        x_tr = np.array(X_train).reshape(np.array(X_train).shape + (1,))
        x_ts = np.array(X_test).reshape(np.array(X_test).shape + (1,))

        # i need to convert labels to one-hot labels (all zeros except for the label
        # for instance: 5 --> 0000010000
        y_tr = to_categorical(np.array(y_train))
        y_ts = to_categorical(np.array(y_test))

        self.model.fit(x_tr, y_tr, validation_data=(x_ts, y_ts), epochs=epochs, verbose=False)

    def predict(self, X):
        prediction = self.model.predict(np.array(X).reshape(np.array(X).shape + (1,)))
        return [np.argmax(x) for x in prediction]  # conversion from one-hot encoding to single label


# implementation of f1_score for metric purpose
# CREDITS: https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

# I had to do it this way because the sklearn f1_score function gave problems with keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
