# consul.py
import consul


class ConsulConfig:
    def __init__(self, url):
        self.client = consul.Consul(url)

    def get_config(self, key):
        return self.client.kv.get(key)[1]["Value"]
