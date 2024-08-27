import tweepy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalysis:
    def __init__(self, api_key, api_secret, access_token, access_token_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth)
        self.sia = SentimentIntensityAnalyzer()

    def get_tweets(self, query, count):
        tweets = tweepy.Cursor(self.api.search, q=query, lang='en', tweet_mode='extended').items(count)
        return [tweet.full_text for tweet in tweets]

    def analyze_sentiment(self, tweets):
        sentiments = []
        for tweet in tweets:
            blob = TextBlob(tweet)
            sentiments.append(self.sia.polarity_scores(tweet)['compound'])
        return sentiments
