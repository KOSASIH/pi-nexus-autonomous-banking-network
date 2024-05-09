import spacy

nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    entities = [(X.text, X.label_) for X in doc.ents]
    return entities

def extract_keywords(text):
    doc = nlp(text)
    keywords = [X.text for X in doc if X.is_stop != True and X.is_punct != True]
    return keywords
