import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class AdvancedNLPChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_text(self, text):
        tokens = word_tokenize(text)
        return tokens

    def lemmatize_text(self, tokens):
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def generate_response(self, text):
        tokens = self.tokenize_text(text)
        lemmatized_tokens = self.lemmatize_text(tokens)
        response = " ".join(lemmatized_tokens)
        return response

# Example usage:
advanced_nlp_chatbot = AdvancedNLPChatbot()
text = "Hello, how are you?"
response = advanced_nlp_chatbot.generate_response(text)
print(response)
