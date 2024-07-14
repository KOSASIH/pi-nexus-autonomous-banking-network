# File name: financial_sentiment_analysis.py
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel

class FinancialSentimentAnalysis:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)

    def analyze(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )
        outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        sentiment = np.argmax(outputs.last_hidden_state[:, 0, :])
        return sentiment

financial_sentiment_analysis = FinancialSentimentAnalysis("bert-base-uncased")
text = "The stock market is going to crash!"
sentiment = financial_sentiment_analysis.analyze(text)
print(sentiment)
