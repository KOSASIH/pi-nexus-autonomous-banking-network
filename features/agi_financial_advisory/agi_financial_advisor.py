import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load financial data
data = pd.read_csv('financial_data.csv')

# Preprocess text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text_data'])

# Train ML model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, data['labels'])

# Load pre-trained NLP model
nlp_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def generate_advice(user_input):
    # Preprocess user input
    input_text = tokenizer.encode_plus(user_input, 
                                        add_special_tokens=True, 
                                        max_length=512, 
                                        return_attention_mask=True, 
                                        return_tensors='pt')
    
    # Get NLP model predictions
    nlp_output = nlp_model(input_text['input_ids'], attention_mask=input_text['attention_mask'])
    sentiment = torch.argmax(nlp_output.logits)
    
    # Get ML model predictions
    ml_output = model.predict(vectorizer.transform([user_input]))
    advice = ml_output[0]
    
    # Combine NLP and ML predictions
    if sentiment == 1:  # Positive sentiment
        advice = 'Consider investing in ' + advice
    elif sentiment == 0:  # Neutral sentiment
        advice = 'Hold onto ' + advice
    else:  # Negative sentiment
        advice = 'Avoid ' + advice
    
    return advice

# Example usage
user_input = 'I want to invest in a high-growth stock.'
print(generate_advice(user_input))
