import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentPredictor:
    def __init__(self, data, target_variable, test_size=0.2, random_state=42):
        self.data = data
        self.target_variable = target_variable
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = []

    def preprocess_data(self):
        # Convert text data to TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=5000)
        self.data['text'] = vectorizer.fit_transform(self.data['text'])

        # Encode target variable using LabelEncoder
        le = LabelEncoder()
        self.data[self.target_variable] = le.fit_transform(self.data[self.target_variable])

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split
