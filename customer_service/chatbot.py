import time

class Chatbot:
    def __init__(self, nlp_model, knowledge_base):
        self.nlp_model = nlp_model
        self.knowledge_base = knowledge_base

    def chat(self, user_input):
        entities = self.nlp_model.extract_entities(user_input)
        intent = self.nlp_model.extract_intent(user_input)
        response = self.knowledge_base.get_response(intent)
        print('User:', user_input)
        print('Chatbot:', response)
        time.sleep(1)
