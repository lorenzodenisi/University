import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import LemmaTokenizer
from nltk.corpus import stopwords as sw
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import sys
import csv
from mpl_toolkits.mplot3d import Axes3D
# from MulticoreTSNE import MulticoreTSNE as TSNE #used only for multicore TSNE on linux
from sklearn.cluster import AgglomerativeClustering

from sklearn.manifold import TSNE


def main():
    print("Loading data")
    data = load_data("T-newsgroups")
    print("Extracting features")
    tfidf_X = feature_extraction(data, 2, 0.03)
    print("Reducing dimensionality")
    reduced_X = reduce_dim(tfidf_X, 3)

    print("Clusterizing")
    clusters = clusterize(reduced_X, 3, 20, tfidf_X)
    plot_distribution(reduced_X, clusters.labels_)

    dump_to_file(clusters.labels_, 'clusters.csv')


def load_data(dirpath):
    files_names = os.listdir(dirpath)
    data = [''] * (len(files_names))

    for filename in files_names:
        sys.stdout.write("\r" + dirpath + '/' + filename)
        f = open(dirpath + '/' + filename, "r")
        data[int(filename)] = f.read()
        f.close()
    print("\n")
    return data


def feature_extraction(data, min_freq, max_freq):
    stop_words = sw.words('english')
    # extending stopwords
    stop_words.extend((
        'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'ha', 'might', 'must', 'need', 'onc', 'onli',
        'ourselv', 'sha', 'themselv', 'veri', 'wa', 'whi', 'wo', 'would', 'yourselv'))

    # extraction of features
    tokenizer = LemmaTokenizer.LemmaTokenizer()
    vect = TfidfVectorizer(tokenizer=tokenizer, encoding='utf-8', strip_accents='unicode',
                           lowercase=True, stop_words=stop_words, max_df=max_freq, min_df=min_freq, sublinear_tf=True)

    numerical = vect.fit_transform(data)
    return numerical


def reduce_dim(data, components):
    # reducing dimensions
    svd = TruncatedSVD(n_components=50, random_state=42)
    tsne = TSNE(n_components=components, perplexity=50, verbose=1, init='random', learning_rate=2000, n_iter=1000,
                early_exaggeration=12, method='exact')

    pipeline = make_pipeline(svd, tsne)
    red_X = pipeline.fit_transform(data)

    # red_X = svd.fit_transform(data)

    return np.array(red_X)


def plot_distribution(data, labels):
    # visualize first 3 dimension in a 3d scatter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=5)
    plt.show()


def clusterize(points, min, max, original):
    # clustering MiniBatchKMeans

    scores = []
    clusters = []
    for i in range(min, max+1):
        print("Trying with " + str(i) + " clusters")
        # km = KMeans(n_clusters=i, precompute_distances=True, verbose=1)
        # clusters.append(km.fit(points))
        minib = MiniBatchKMeans(n_clusters=i, init_size=1024, batch_size=1024, random_state=20)
        clusters.append(minib.fit(points))
        # agg = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
        # clusters.append(agg.fit(points))
        scores.append(silhouette_score(points, clusters[i - min].labels_))

    plt.plot(range(min, max+1), scores)  # plotting silhouette values for all numbers of clusters
    plt.show()
    return clusters[np.argmax(scores)+3]  # I pick the best


def dump_to_file(clusters, filename):
    """Dump the evaluated labels to a CSV file."""

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Id', 'Predicted'])

        for n, label in enumerate(clusters):
            writer.writerow([n, label])


if __name__ == "__main__":
    main()
