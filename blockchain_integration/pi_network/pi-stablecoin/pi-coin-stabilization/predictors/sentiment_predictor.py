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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['text'], self.data[self.target_variable], test_size=self.test_size, random_state=self.random_state)

    def train_models(self):
        # Create and train multiple models
        models = [
            MultinomialNB(),
            LogisticRegression(max_iter=1000),
            SVC(kernel='linear', C=1),
            RandomForestClassifier(n_estimators=100, max_depth=5)
        ]

        for model in models:
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

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

    def make_predictions(self, input_text):
        # Make predictions using the best model
        best_model = max(self.models, key=lambda x: x.score(self.X_test, self.y_test))
        input_vector = self.vectorizer.transform([input_text])
        prediction = best_model.predict(input_vector)

        return prediction

    def train_transformer_model(self):
        # Train a transformer model using Hugging Face's AutoModelForSequenceClassification
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        # Prepare dataset for training
        train_dataset = self.data[['text', self.target_variable]]
        train_dataset['text'] = train_dataset['text'].apply(lambda x: tokenizer.encode_plus(x, 
                                                                                             add_special_tokens=True, 
                                                                                             max_length=512, 
                                                                                             return_attention_mask=True, 
                                                                                             return_tensors='pt'))

        # Train the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in train_dataset:
                input_ids = batch['text']['input_ids'].to(device)
                attention_mask = batch['text']['attention_mask'].to(device)
                labels = batch[self.target_variable].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataset)}')

        self.transformer_model = model

    def make_transformer_predictions(self, input_text):
        # Make predictions using the transformer model
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        input_ids = tokenizer.encode_plus(input_text, 
                                           add_special_tokens=True, 
                                           max_length=512, 
                                           return_attention_mask=True, 
                                           return_tensors='pt')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = input_ids['input_ids'].to(device)
        attention_mask = input_ids['attention_mask'].to(device)

        self.transformer_model.eval()
        with torch.no_grad():
            outputs = self.transformer_model(input_ids, attention_mask=attention_mask)

        prediction = torch.argmax(outputs.logits)

        return prediction
