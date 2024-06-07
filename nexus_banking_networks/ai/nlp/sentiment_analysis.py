import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    def __init__(self, num_classes):
        super(SentimentAnalyzer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.fc(pooled_output)
        return x

class SentimentAnalysisSystem:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer

    def analyze_sentiment(self, text):
        inputs = self.sentiment_analyzer.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        output = self.sentiment_analyzer(input_ids, attention_mask)
        return output
