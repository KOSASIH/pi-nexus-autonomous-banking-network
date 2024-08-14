from .node_software import NodeSoftware
from .node_config import NodeConfig

class Node:
    def __init__(self, node_id, software, config):
        self.node_id = node_id
        self.software = software
        self.config = config

    def start(self):
        self.software.start()
        self.config.apply()

    def stop(self):
        self.software.stop()
        self.config.revert()

    def update_software(self, new_software):
        self.software = new_software
        self.start()

    def update_config(self, new_config):
        self.config = new_config
        self.start()
