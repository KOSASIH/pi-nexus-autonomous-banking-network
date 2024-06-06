import nltk
from nltk.chat.util import Chat, reflections

pairs = [
    [
        r"Hi|Hello|Hey",
        ["Hello! How can I help you today?", "Hi there!", "Hey! What's up?"]
    ],
    [
        r"What's the balance of my account\?",
        ["Your account balance is $1234.56. Would you like to perform any transactions?",]
    ],
    # Add more pairs for different customer queries
]

def main():
    print("Welcome to Nexus Banking Network AI-powered Customer Support!")
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__ == "__main__":
    main()
