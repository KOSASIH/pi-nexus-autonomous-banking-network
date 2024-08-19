import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FeatureEngineer:
    def __init__(self, data, target_variable):
        self.data = data
        self.target_variable = target_variable

    def preprocess_text(self, text_data):
        # Convert text data to lowercase
        text_data = text_data.apply(lambda x: x.lower())

        # Remove punctuation and special characters
        text_data = text_data.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

        return text_data

    def extract_tfidf_features(self, text_data):
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_features = vectorizer.fit_transform(text_data)

        return tfidf_features

    def extract_bert_features(self, text_data, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        features = []
        for text in text_data:
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            features.append(outputs.last_hidden_state[:, 0, :].detach().numpy())

        features = np.array(features)

        return features

    def scale_features(self, features):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        return scaled_features

    def engineer_features(self, text_data, model_name):
        text_data = self.preprocess_text(text_data)
        tfidf_features = self.extract_tfidf_features(text_data)
        bert_features = self.extract_bert_features(text_data, model_name)
        features = np.concatenate((tfidf_features.toarray(), bert_features), axis=1)
        scaled_features = self.scale_features(features)

        return scaled_features
