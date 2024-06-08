import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Load dataset
dataset = pd.read_csv('customer_feedback.csv')

# Create a SentimentIntensityAnalyzer instance
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each feedback
sentiments = []
for feedback in dataset['feedback']:
    tokens = word_tokenize(feedback)
    tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]
    sentiment = sia.polarity_scores(' '.join(tokens))
    sentiments.append(sentiment)

# Calculate overall sentiment
overall_sentiment = Counter([sentiment['compound'] for sentiment in sentiments])

print("Overall sentiment:", overall_sentiment)
