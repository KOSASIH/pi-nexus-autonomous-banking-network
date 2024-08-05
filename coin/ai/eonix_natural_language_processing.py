import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import re
import string

class EonixNLP:
    def __init__(self):
        self.vectorizer = None
        self.model = None

    def load_data(self, df, text_column, label_column):
        self.df = df
        self.text_column = text_column
        self.label_column = label_column

    def preprocess_text(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: x.lower())
        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: re.sub(r'\d+', '', x))
        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: re.sub(r'[{}]'.format(string.punctuation), '', x))
        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x) if word not in stop_words]))

    def vectorize_text(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(self.df[self.text_column])
        y = self.df[self.label_column]
        return X, y

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict_text(self, text):
        text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([text])
        return self.model.predict(text_vector)[0]
