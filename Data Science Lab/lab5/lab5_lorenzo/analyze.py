import os
import sys
import csv
import nltk
from nltk.corpus import stopwords as sw
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mlxtend.frequent_patterns import apriori
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import re


def main():
    data = load_data("T-newsgroups")
    clusters = load_clusters(data, "best_clusters.csv")

    stop_words = sw.words('english')
    stop_words.extend((
        'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'ha', 'might', 'must', 'need', 'onc', 'onli',
        'ourselv', 'sha', 'themselv', 'veri', 'wa', 'whi', 'wo', 'would', 'yourselv'))

    freq = []
    mask = np.array(Image.open("space.png"))
    wc = WordCloud(stopwords=stop_words, width=1500, height=1500, max_words=1000, background_color='white')
    total_text = ""
    for cluster in clusters:
        clusterseq = " ".join(cluster)
        freq.append(text_to_freq(clusterseq, stop_words))  # evaluating term frequencies
        total_text = total_text + " " + clusterseq

    total_freq = text_to_freq(total_text, stop_words)  # evaluating total term frequencies
    max_tf = 500

    # I remove all terms that appears too often through all the documents (in more than 500 documents for instance)
    i = 0
    commons = []
    for key in total_freq.keys():
        if int(total_freq[key]) > max_tf:
            commons.append(key)
            i += 1
            for cluster_freq in freq:
                if key in cluster_freq.keys():  # if a common term is present in the cluster term frequencies, remove it
                    cluster_freq.pop(key)
    print("Pruned " + str(i) + ' words')

    """for n, cluster in enumerate(clusters):
        set = text_to_set(cluster, stop_words, commons)
        te = TransactionEncoder()
        te_ary = te.fit(set).transform(set)

        set = pd.DataFrame(te_ary, columns=te.columns_)
        result = apriori(set, min_support=0.08, use_colnames=True)
        result = sorted(result.values, key=lambda x: -len(x[1]))
        print("Most significant sequences (Cluster"+str(n)+")")
        for res in result[:10]:
            print(res[1])
"""
    images = []
    for f in freq:
        wc.generate_from_frequencies(f)
        images.append(wc.to_image())

    # visualizing all clusters in a grid

    rows = int(np.ceil(np.sqrt(len(clusters))))
    cols = int(len(clusters) / rows)

    fig = plt.figure(dpi=2000)
    for row in range(rows):
        for col in range(cols):
            fig.add_subplot(rows, cols, col + row * cols + 1)
            plt.axis('off')
            plt.imshow(images[col + row * cols])

    plt.savefig("wordcloud.jpg", dpi=2000)
    plt.show()

    for n, image in enumerate(images):
        plt.axis('off')
        plt.imshow(image)
        plt.savefig("cluster" + str(n) + ".jpg", dpi=2000)


def text_to_freq(raw_text, stopWords):
    raw_text = raw_text.lower()  # lowercase
    raw_text = raw_text.replace("'", "")
    tokens = nltk.word_tokenize(raw_text)  # tokenize
    text = nltk.Text(tokens)

    text_content = [word for word in text if word not in stopWords]  # removing stopwords
    text_content = [s for s in text_content if len(s) != 0]
    WNL = nltk.WordNetLemmatizer()  # apply lemmatizing
    text_content = [WNL.lemmatize(t) for t in text_content]
    fdist = nltk.FreqDist(text_content)  # evaluate frequencies

    word_dict = {}

    for key in fdist:
        word_dict[key] = fdist[key]

    return word_dict


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


def load_clusters(data, csv_path):
    ids = {
        "Id": [],
        "Predicted": []
    }
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        header = True
        for row in csv_reader:
            if header:
                header = False
                continue

            ids["Id"].append(row[0])
            ids["Predicted"].append(row[1])

    n_clusters = int(max(ids["Predicted"])) + 1  # eval total number of clusters
    clusters = [[] for i in range(n_clusters)]
    for id, cluster in zip(ids["Id"], ids["Predicted"]):
        clusters[int(cluster)].append(data[int(id)])  # appending all text belonging to the same cluster

    return clusters


def text_to_set(docs, stopwords, commons):
    sets = []
    for doc in docs:
        regex = re.compile("^[a-zA-Z0-9]+$")

        doc = doc.lower()  # lowercase
        doc = doc.replace("'", "")
        tokens = nltk.word_tokenize(doc)  # tokenize
        text = nltk.Text(tokens)

        text_content = [word for word in text if
                        (word not in stopwords and word not in commons and regex.match(word))]  # removing stopwords
        text_content = [s for s in text_content if len(s) > 2]
        WNL = nltk.WordNetLemmatizer()  # apply lemmatizing
        text_content = [WNL.lemmatize(t) for t in text_content]
        sets.append(text_content)
    return sets


if __name__ == "__main__":
    main()
