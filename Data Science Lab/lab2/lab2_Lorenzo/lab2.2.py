import csv
import string
import math


def main():
    docs = []
    rate = []

    with open('aclimdb_reviews_train.txt', encoding='utf8') as f:
        skip = False
        for cols in csv.reader(f):
            if not skip:
                skip = True
                continue

            docs.append(cols[0])
            rate.append(cols[1])

    tokens = tokenize(docs)

    tf = []

    for token_row in tokens:
        doc_tf = {}
        for tok in token_row:
            if tok not in doc_tf.keys():
                doc_tf[tok] = 1
            else:
                doc_tf[tok] += 1
        tf.append(doc_tf)

    df = DF(tf)
    idf = IDF(df, len(docs), sorted=False)
    tf_idf = TF_IDF(tf, idf)

    goods = []
    bads = []

    for i in range(len(tf_idf)):
        if rate[i] == '1':
            goods.append(tf_idf[i])
        else:
            bads.append(tf_idf[i])

    right = 0
    for i in range(10):
        if evaluate(tf_idf[i], goods, bads) == int(rate[i]):
            right += 1

    print(str(right)+'/10')


def tokenize(docs):
    """Compute the tokens for each document.
    Input: a list of strings. Each item is a document to tokenize.
    Output: a list of lists. Each item is a list containing the tokens of the
    relative document.
    """

    tokens = []
    for doc in docs:
        for punct in string.punctuation:
            doc = doc.replace(punct, " ")
        split_doc = [token.lower() for token in doc.split(" ") if token]
        tokens.append(split_doc)
    return tokens


def DF(data):
    de = {}
    for row in data:
        for word in row.keys():
            if word not in de.keys():
                de[word] = 1
            else:
                de[word] += 1
    return de


def IDF(df, N, sorted=False):
    idf = []
    for word, freq in zip(df.keys(), df.values()):
        idf.append((word, math.log(N / freq)))

    idf_dict = {}

    if sorted:
        idf.sort(key=lambda x: x[0])

    for word, _idf in idf:
        idf_dict[word] = _idf

    return idf_dict


def TF_IDF(tf, idf):
    tf_idf = []
    for doc in tf:
        doc_tokens = {}
        for tok, _tf in zip(doc.keys(), doc.values()):
            doc_tokens[tok] = _tf * idf[tok]
        tf_idf.append(doc_tokens)

    return tf_idf


def evaluate(v1, goods, bads):
    good_val = 0
    bad_val = 0
    for good in goods:
        good_val += cosine_similarity(v1, good)

    for bad in bads:
        bad_val += cosine_similarity(v1, bad)

    good_conf = good_val / (good_val + bad_val)
    bad_conf = 1 - good_conf

    print("Sentiment analisys:")
    print("Good: " + str(good_conf) + ", Bad: " + str(bad_conf))

    return 1 if good_conf > bad_conf else 0


def norm(d):
    """Compute the L2-norm of a vector representation."""
    return sum([tf_idf ** 2 for t, tf_idf in d.items()]) ** .5


def dot_product(d1, d2):
    """Compute the dot product between two vector representations."""
    word_set = set(list(d1.keys()) + list(d2.keys()))
    return sum([(d1.get(d, 0.0) * d2.get(d, 0.0)) for d in word_set])


def cosine_similarity(d1, d2):
    """
    Compute the cosine similarity between documents d1 and d2.
    Input: two dictionaries representing the TF-IDF vectors for documents
    d1 and d2.
    Output: the cosine similarity.
    """
    return dot_product(d1, d2) / (norm(d1) * norm(d2))


if __name__ == "__main__":
    main()
