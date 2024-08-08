import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from utils.model_utils import train_model, evaluate_model

def train_models():
    # Load processed data
    crypto_data = pd.read_csv('processed_crypto_data.csv')
    weather_data = pd.read_json('processed_weather_data.json')
    social_media_data = pd.read_json('processed_social_media_data.json')

    # Train models
    crypto_model = train_model(crypto_data, MLPClassifier())
    weather_model = train_model(weather_data, RandomForestClassifier())
    social_media_model = train_model(social_media_data, MultinomialNB())

    # Evaluate models
    evaluate_model(crypto_model, crypto_data)
    evaluate_model(weather_model, weather_data)
    evaluate_model(social_media_model, social_media_data)

    # Save trained models to disk
    crypto_model.save('crypto_market_predictor.h5')
    weather_model.save('weather_forecaster.pkl')
    social_media_model.save('sentiment_analyzer.joblib')

if __name__ == '__main__':
    train_models()
