import rasa

class ARChatbot:
    def __init__(self, model_path):
        self.model_path = model_path
        self.agent = rasa.core.agent.load(model_path)

    def respond_to_user_queries(self, message):
        # Respond to user queries
        response = self.agent.handle_message(message)
        return response

class AdvancedARChatbot:
    def __init__(self, ar_chatbot):
        self.ar_chatbot = ar_chatbot

    def enable_conversational_banking(self, message):
        # Enable conversational banking
        response = self.ar_chatbot.respond_to_user_queries(message)
        return response
