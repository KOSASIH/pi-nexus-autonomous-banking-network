import nltk
from nltk.stem import LancasterStemmer
from sklearn.naive_bayes import MultinomialNB

# Load chatbot model
chatbot_model = nltk.load('nlp_models/chatbot_model.pkl')

# Define ML chatbot function
def respond_to_user_input(user_input):
    # Preprocess user input
    stemmed_input = LancasterStemmer().stem(user_input)
    
    # Classify user input using chatbot model
    classification = chatbot_model.classify(stemmed_input)
    
    # Return response based on classification
    if classification == 'hello':
        return 'Hello! How can I assist you today?'
    elif classification == 'help':
        return 'I\'m here to help. What do you need assistance with?'
    else:
        return 'I didn\'t understand that. Can you please rephrase?'
