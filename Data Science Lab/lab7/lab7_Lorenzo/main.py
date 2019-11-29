import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.stats as stats
from scipy import signal
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from librosa.effects import trim
import csv
import conv1d
import NN_random_forest
import seaborn as sns

# PARAMETERS
TRIM_THRESHOLD = 10
N_TREES = 10
TRAIN_SIZE = 0.80  # ratio between train and validation (holdout)
SAMPLE_RATE = 8000  # we assume fixed sample rate


# N_CHUNK = 50

def main():
    data, eval = load_data("free-spoken-digit/dev", "free-spoken-digit/eval")

    # preprocessing of training data
    processed = preprocessing(data["data"])
    # preprocessing of unlabeled data
    eval_processed = preprocessing(eval["data"], len(processed[0]))

    # prediction by random forest of neural networks
    predictions = train_predict_spectrogram(processed, data["label"], eval_processed)

    dump_to_file(predictions, eval["id"], "labels.csv")


def load_data(dev_path, eval_path):
    dev_data = {
        "data": [],
        "id": [],
        "label": []
    }
    eval_data = {
        "data": [],
        "id": [],
    }

    dev_files = os.listdir(dev_path)
    eval_files = os.listdir(eval_path)

    for file in dev_files:
        dev_data["data"].append(wavfile.read(dev_path + "/" + file, True))
        name, label, _ = re.split('[_.]', file)  # split file name to id and label
        dev_data["label"].append(label)
        dev_data["id"].append(name)

    for file in eval_files:
        eval_data["data"].append(wavfile.read(eval_path + "/" + file, True))
        name, _ = re.split('[.]', file)
        eval_data["id"].append(name)

    return dev_data, eval_data


def preprocessing(data, length=None):
    records = [np.array(x[1], dtype=float) for x in data]  # every record has also the sample rate as first element

    # print("Normalizing")  # normalization dividing by the local max value
    # normalized_data = normalization(records)
    print("Trimming")  # removing silences
    trimmed = trim_data(records, TRIM_THRESHOLD)
    print("Padding")  # zero padding to uniform lengths
    padded = zero_padding(trimmed, length)
    return padded


def normalization(data):
    normalized = []
    for record in data:
        normalized.append(np.array(record) / np.max(record))
        a# normalized.append(np.array(record, dtype=float))
    return normalized


def trim_data(data, threshold):
    trimmed = []
    for record in data:
        trimmed.append((trim(record, top_db=threshold))[0])  # trim function from librosa
    return trimmed


def zero_padding(data, final_len=None):
    padded = []
    if final_len is None:
        final_len = int(np.max([len(x) for x in data]))

    for record in data:
        to_add = final_len - len(record)
        p = np.pad(np.array(record), (0, to_add), constant_values=0)  # append zeros to the end
        padded.append(p)

    return padded


def train_predict_spectrogram(X, y, to_eval):
    # compute spectrograms for each record
    spectrograms = get_spectrogram(X)
    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(spectrograms, y, train_size=TRAIN_SIZE)

    # init nn random forest
    model = NN_random_forest.NNRandomForest(N_TREES,
                                            input_shape=(spectrograms[0].shape[0], spectrograms[0].shape[1], 1),
                                            output_size=10)
    # build the models
    model.build()

    # train the models
    model.fit(X_train, X_test, y_train, y_test)
    # print metrics summary for test data
    print(classification_report(np.array(y_test, dtype=int), model.predict(X_test)))
    m = confusion_matrix(np.array(y_test, dtype=int), model.predict(X_test))
    sns.heatmap(m, annot=True, cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    # compute predictions for unlabeled data
    to_eval_spectrogram = get_spectrogram(to_eval)
    predictions = model.predict(to_eval_spectrogram)
    return predictions


def get_spectrogram(signals):
    spec = []
    eps = 1
    for sig in signals:
        frequencies, times, s = signal.spectrogram(sig, SAMPLE_RATE, scaling='spectrum')  # compute spectrogram
        spec.append(np.log10(s + eps))  # I need to sum an epsilon to avoid log10(0) = Inf
        #plt.pcolormesh(times, frequencies, np.log10(s+eps))
        #plt.colorbar()
        #plt.xlabel("Time (s)")
        #plt.ylabel("Frequency (Hz)")
        #plt.title("Spectrogram")
        #plt.tight_layout()
        #plt.show()
    return spec


def dump_to_file(prediction, ids, filename):
    """Dump the evaluated labels to a CSV file."""

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Id', 'Predicted'])

        for n, label in enumerate(prediction):
            writer.writerow([ids[n], label])


'''
# NOT USED
def transformation(data):
    print("Transform")
    spectrums = get_spectrum(data, 0, SAMPLE_RATE / 2)
    print("Evaluating spectral density")
    spectral_density = get_spectral_density(spectrums)
    return spectral_density


# NOT USED
def feature_extraction(data):
    print("Chunking")
    chunks = chunker(data, N_CHUNK)
    print("Extracting features")
    features = get_features(chunks)
    return features


#  NOT USED
def train_predict_NN(X, y, to_eval):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)

    model = conv1d.conv1D((len(X_train[0]), 1), 10)

    conv1d.fit_conv1d(model, X_train, np.array(X_test), y_train, np.array(y_test))
    predictions = conv1d.predict_conv1d(model, to_eval)
    return predictions


# NOT USED
def subsample(data):
    processed = []
    final_len = int(np.max([len(x) for x in data]))
    for n, record in enumerate(data):
        sample_indexes = sample(len(record), final_len)
        processed.append(record[sample_indexes])

    return processed


# NOT USED
def sample(start_len, final_len):
    ratio = float(start_len) / final_len
    indexes = []
    i = 0
    while len(indexes) < final_len:
        if int(np.around(i)) == start_len:
            i -= 1

        indexes.append(int(np.around(i)))
        i += ratio

    return np.array(indexes, dtype=int)


# NOT USED
def chunker(data, n_chunk):
    chunks = []
    for d in data:
        data_chunks = []
        dim = int(np.floor(len(d) / n_chunk))
        for i in range(n_chunk):
            min = i * dim
            max = np.min([(i + 1) * dim, len(d) - 1])
            data_chunks.append(d[min:max])
        chunks.append(data_chunks)

    return chunks


# NOT USED
def get_spectrum(records, min_freq, max_freq, sr=SAMPLE_RATE):
    spectrums = []
    # total length of record corresponds to a bandwidth of 8k, mirrored
    # so we take the first part and apply the low pass and high pass filter

    for record in records:
        min = int((min_freq / (sr / 2)) * (len(record) / 2))
        max = int((max_freq / (sr / 2)) * (len(record) / 2))
        spectrums.append((np.array(fft(record)))[min:max])
    return spectrums


# NOT USED
def get_features(spectral_density):
    features = []
    i = 0
    for sd in spectral_density:
        f = []
        for chunk in sd:
            f.append(np.mean(chunk))
            f.append(np.var(chunk))
            f.append(stats.skew(chunk))
            # f.append(stats.kurtosis(chunk))
        features.append(f)
        i += 1
    return features


# NOT USED
def get_spectral_density(spectrums):
    sd = []
    for spectrum in spectrums:
        density = np.power(np.abs(spectrum), 2)
        sd.append(density)
    return sd


# NOT USED
def train(X, y, n_trees, max_features):
    rf = RandomForestClassifier(n_trees, max_features=max_features, warm_start=True, verbose=True,
                                criterion="gini")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print(f1_score(y_test, y_pred, average='macro'))
    return rf

'''
if __name__ == "__main__":
    main()
