import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class NaturalLanguageProcessor:
    def __init__(self, text):
        self.text = text
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def process(self):
        tokens = word_tokenize(self.text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return filtered_tokens

# Example usage
if __name__ == "__main__":
    text = "This is an example sentence for natural language processing."
    processor = NaturalLanguageProcessor(text)
    processed_text = processor.process()
    print("Processed Text:", processed_text)
