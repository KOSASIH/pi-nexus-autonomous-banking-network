import pandas as pd
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self, data):
        self.data = data

    def analyze_sentiment(self):
        self.data['sentiment'] = self.data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        return self.data[['text', 'sentiment']]

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({'text': ['I love this!', 'This is terrible.', 'I am neutral.']})
    analyzer = SentimentAnalyzer(data)
    results = analyzer.analyze_sentiment()
    print("Sentiment Analysis Results:\n", results)
