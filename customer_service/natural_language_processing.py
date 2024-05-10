import spacy

nlp = spacy.load("en_core_web_sm")


class NLP:
    def __init__(self):
        self.nlp = nlp

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = [(X.text, X.label_) for X in doc.ents]
        return entities

    def extract_intent(self, text):
        doc = self.nlp(text)
        intents = {"account": False, "transaction": False, "other": False}
        for token in doc:
            if token.text.lower() in ["account", "accounts"]:
                intents["account"] = True
            elif token.text.lower() in ["transaction", "transactions"]:
                intents["transaction"] = True
        if any(intents.values()):
            return intents
        else:
            intents["other"] = True
            return intents
