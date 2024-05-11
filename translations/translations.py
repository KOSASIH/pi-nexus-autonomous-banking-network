import os


def load_translations():
    # Load all available translations
    translations_dir = os.path.join(os.path.dirname(__file__), "translations")
    translations = {}
    for file in os.listdir(translations_dir):
        if file.endswith(".json"):
            with open(os.path.join(translations_dir, file), "r") as f:
                language = file[:-5]
                translations[language] = json.load(f)
    return translations


translations = load_translations()
