import real_time_sentiment_analysis

# Define a real-time sentiment analysis model
def real_time_sentiment_analysis_model():
    model = real_time_sentiment_analysis.RealTimeSentimentAnalysisModel()
    return model

# Use the real-time sentiment analysis model to analyze customer feedback
def analyze_feedback(model, feedback):
    sentiment = model.analyze_feedback(feedback)
    return sentiment
