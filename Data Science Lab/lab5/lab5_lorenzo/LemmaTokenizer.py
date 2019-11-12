from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re


class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")

    def __call__(self, documents):
        lemmas = []
        regex = re.compile("^[a-zA-Z.0-9@]+$")
        for t in word_tokenize(documents):
            t = t.strip()
            t = self.stemmer.stem(t)
            lemma = self.lemmatizer.lemmatize(t)

            x = regex.match(lemma)
            if x:
                lemmas.append(lemma)
        return lemmas
