# tokenization.py

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def tokenize(text):
    return word_tokenize(text)


# stemming.py


def stem(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]
