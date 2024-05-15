# tokenization.py

import nltk
from nltk.tokenize import word_tokenize

def tokenize(text):
    return word_tokenize(text)

# stemming.py

import nltk
from nltk.stem import PorterStemmer

def stem(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]
