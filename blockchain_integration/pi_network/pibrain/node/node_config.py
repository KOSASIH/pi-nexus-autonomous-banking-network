# node_config.py

import json
import os
import logging
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

class NodeConfig:
    """Node configuration class."""

    def __init__(self, node_id: str, node_name: str, node_type: str, node_address: str, node_port: int):
        self.node_id = node_id
        self.node_name = node_name
        self.node_type = node_type
        self.node_address = node_address
        self.node_port = node_port
        self.config_data = {}

    def load_config(self, config_file: str) -> None:
        """Load node configuration from a file."""
        with open(config_file, 'r') as f:
            self.config_data = json.load(f)

    def save_config(self, config_file: str) -> None:
        """Save node configuration to a file."""
        with open(config_file, 'w') as f:
            json.dump(self.config_data, f, indent=4)

    def get_config(self, key: str) -> Any:
        """Get a configuration value by key."""
        return self.config_data.get(key)

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value by key."""
        self.config_data[key] = value

    def get_node_id(self) -> str:
        """Get the node ID."""
        return self.node_id

    def get_node_name(self) -> str:
        """Get the node name."""
        return self.node_name

    def get_node_type(self) -> str:
        """Get the node type."""
        return self.node_type

    def get_node_address(self) -> str:
        """Get the node address."""
        return self.node_address

    def get_node_port(self) -> int:
        """Get the node port."""
        return self.node_port

class NodeConfigManager:
    """Node configuration manager class."""

    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.node_configs = {}

    def load_node_configs(self) -> None:
        """Load all node configurations from files."""
        for file in os.listdir(self.config_dir):
            if file.endswith('.json'):
                node_id = file.split('.')[0]
                config_file = os.path.join(self.config_dir, file)
                node_config = NodeConfig(node_id, '', '', '', 0)
                node_config.load_config(config_file)
                self.node_configs[node_id] = node_config

    def get_node_config(self, node_id: str) -> Optional[NodeConfig]:
        """Get a node configuration by ID."""
        return self.node_configs.get(node_id)

    def save_node_configs(self) -> None:
        """Save all node configurations to files."""
        for node_config in self.node_configs.values():
            config_file = os.path.join(self.config_dir, f'{node_config.node_id}.json')
            node_config.save_config(config_file)

def main():
    logging.basicConfig(level=logging.INFO)
    config_dir = 'configs'
    node_config_manager = NodeConfigManager(config_dir)
    node_config_manager.load_node_configs()

    node_id = 'node-1'
    node_config = node_config_manager.get_node_config(node_id)
    if node_config:
        print(f'Node ID: {node_config.get_node_id()}')
        print(f'Node Name: {node_config.get_node_name()}')
        print(f'Node Type: {node_config.get_node_type()}')
        print(f'Node Address: {node_config.get_node_address()}')
        print(f'Node Port: {node_config.get_node_port()}')
    else:
        print(f'Node config not found: {node_id}')

if __name__ == '__main__':
    main()
