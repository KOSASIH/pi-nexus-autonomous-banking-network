import pandas as pd
import matplotlib.pyplot as plt
from utils.data_utils import load_data, process_data
from utils.visualization_utils import visualize_data

def analyze_data():
    # Load processed data
    crypto_data = pd.read_csv('processed_crypto_data.csv')
    weather_data = pd.read_json('processed_weather_data.json')
    social_media_data = pd.read_json('processed_social_media_data.json')

    # Perform advanced data analysis
    crypto_analysis = crypto_data.groupby('date').agg({'price': 'mean'}).reset_index()
    weather_analysis = weather_data.groupby('location').agg({'temperature': 'mean'}).reset_index()
    social_media_analysis = social_media_data.groupby('platform').agg({'sentiment': 'mean'}).reset_index()

    # Visualize data
    visualize_data(crypto_analysis, 'Crypto Market Analysis')
    visualize_data(weather_analysis, 'Weather Analysis')
    visualize_data(social_media_analysis, 'Social Media Sentiment Analysis')

if __name__ == '__main__':
    analyze_data()
