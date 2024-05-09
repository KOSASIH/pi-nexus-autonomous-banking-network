import unittest
import spacy
from ai.natural_language_processing import extract_entities, extract_keywords

class TestNaturalLanguageProcessing(unittest.TestCase):
    def test_extract_entities(self):
        nlp = spacy.load('en_core_web_sm')
        text = "Apple is a company that makes iPhones."
        entities = extract_entities(text, nlp)
        self.assertIsInstance(entities, list)
        self.assertGreaterEqual(len(entities), 1)

    def test_extract_keywords(self):
        nlp = spacy.load('en_core_web_sm')
        text = "Artificial intelligence is a field of computer science that focuses on creating intelligent machines that can think and learn."
        keywords = extract_keywords(text, nlp)
        self.assertIsInstance(keywords, list)
        self.assertGreaterEqual(len(keywords), 5)

if __name__ == '__main__':
    unittest.main()
