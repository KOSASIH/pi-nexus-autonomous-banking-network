import os
import json
from cosmos_sdk.client.lcd import LCDClient
from tendermint.tendermint import Tendermint

class ChainBridge:
    def __init__(self, chain_ids, lcd_clients):
        self.chain_ids = chain_ids
        self.lcd_clients = lcd_clients
        self.tendermint = Tendermint()

    def get_chain_data(self, chain_id, query):
        lcd_client = self.lcd_clients[chain_id]
        response = lcd_client.query(query)
        return response

    def send_cross_chain_transaction(self, from_chain_id, to_chain_id, tx_data):
        from_lcd_client = self.lcd_clients[from_chain_id]
        to_lcd_client = self.lcd_clients[to_chain_id]
        tx_hash = from_lcd_client.broadcast_tx(tx_data)
        self.tendermint.relay_tx(tx_hash, to_chain_id)

    def start_oracle_service(self):
        while True:
            for chain_id in self.chain_ids:
                data = self.get_chain_data(chain_id, "custom/query")
                if data:
                    self.send_cross_chain_transaction(chain_id, "another_chain_id", data)

if __name__ == "__main__":
    chain_ids = ["cosmoshub-4", "bnb-smart-chain"]
    lcd_clients = {
        "cosmoshub-4": LCDClient("https://lcd.cosmoshub-4.net", "cosmoshub-4"),
        "bnb-smart-chain": LCDClient("https://lcd.bnb-smart-chain.net", "bnb-smart-chain")
    }
    bridge = ChainBridge(chain_ids, lcd_clients)
    bridge.start_oracle_service()
