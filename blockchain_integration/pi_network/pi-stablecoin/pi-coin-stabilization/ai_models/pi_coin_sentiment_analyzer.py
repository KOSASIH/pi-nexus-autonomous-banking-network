import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class PiCoinSentimentAnalyzer:
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
        # Convert text data to lowercase
        self.data['text'] = self.data['text'].apply(lambda x: x.lower())

        # Remove punctuation and special characters
        self.data['text'] = self.data['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

        # Tokenize text data
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.data['text'] = self.data['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['text'],
                                                                              self.data[self.target_variable],
                                                                              test_size=self.test_size,
                                                                              random_state=self.random_state)

    def train_models(self):
        # Create and train multiple models
        models = [
            Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', LogisticRegression())
            ]),
            Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', SVC())
            ]),
            Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', RandomForestClassifier())
            ]),
            Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', MultinomialNB())
            ]),
            self._train_bert_model()
        ]

        for model in models:
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

    def _train_bert_model(self):
        # Train a BERT model
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in self.X_train:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {total_loss / len(self.X_train)}')

        model.eval()
        return model

    def evaluate_models(self):
        # Evaluate each model using accuracy score, classification report, and confusion matrix
        results = []
        for model in self.models:
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            matrix = confusion_matrix(self.y_test, y_pred)
            results.append((model.__class__.__name__, accuracy, report, matrix))

        return results

    def make_predictions(self, input_data):
        # Make predictions using the best model
        best_model = max(self.models, key=lambda x: x.score(self.X_test, self.y_test))
        predictions = best_model.predict(input_data)
        return predictions
