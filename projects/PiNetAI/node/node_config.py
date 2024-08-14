import json

class NodeConfig:
    def __init__(self, config_id, settings):
        self.config_id = config_id
        self.settings = settings

    def apply(self):
        with open(f'/etc/{self.config_id}.json', 'w') as f:
            json.dump(self.settings, f)

    def revert(self):
        with open(f'/etc/{self.config_id}.json', 'r') as f:
            self.settings = json.load(f)
