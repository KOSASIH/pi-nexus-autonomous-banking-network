# utils/helpers.py
def get_config():
    # Read config file and return configuration dictionary
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config
