import json
import os


class Translator:
    def __init__(self, language):
        # Load the translation data for the specified language
        self.language = language
        translations_file = os.path.join(
            os.path.dirname(__file__), "translations", f"{language}.json"
        )
        with open(translations_file, "r") as f:
            self.translations = json.load(f)

    def translate(self, key):
        # Translate a given key into the specified language
        return self.translations.get(key, key)
