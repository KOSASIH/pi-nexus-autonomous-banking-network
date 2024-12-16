import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

class NLP:
    def __init__(self, model_path='nlp_model.pkl'):
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer()
        self.model = None

    @staticmethod
    def load_data(file_path):
        """Load textual data from a CSV file."""
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        """Preprocess the textual data."""
        X = self.vectorizer.fit_transform(data['text'])
        y = data['label']
        return X, y

    def train_model(self, X, y):
        """Train the NLP model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Save the model and vectorizer
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, 'vectorizer.pkl')

    def load_model(self):
        """Load the trained model from a file."""
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load('vectorizer.pkl')

    def predict(self, text):
        """Predict the label of the given text."""
        if self.model is None:
            raise Exception("Model not loaded. Please load the model first.")
        text_vectorized = self.vectorizer.transform([text])
        return self.model.predict(text_vectorized)
