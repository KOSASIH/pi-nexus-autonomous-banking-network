# pi_nexus/nlp.py
import spacy


class NLPModel:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def process_text(self, text: str) -> dict:
        doc = self.nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        return {"entities": entities}
