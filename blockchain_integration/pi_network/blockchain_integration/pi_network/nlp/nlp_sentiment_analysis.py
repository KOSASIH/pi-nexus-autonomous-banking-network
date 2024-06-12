import transformers
from transformers import pipeline

class NLPSentimentAnalysis:
    def __init__(self, model_name):
        self.model = pipeline('sentiment-analysis', model=model_name)

    def analyze_sentiment(self, text):
        result = self.model(text)
        return result

# Example usage:
nlp_sentiment_analyzer = NLPSentimentAnalysis('distilbert-base-uncased-finetuned-sst-2-english')
text = 'I love the new PI-Nexus Autonomous Banking Network!'
result = nlp_sentiment_analyzer.analyze_sentiment(text)
print(result)
