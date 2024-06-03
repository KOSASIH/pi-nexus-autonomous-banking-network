# pi_data_analytics.py

import web3
from web3.contract import Contract


class PIDataAnalytics:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI Data Analytics ABI from a file or database
        with open("pi_data_analytics.abi", "r") as f:
            return json.load(f)

    def get_network_usage(self) -> dict:
        # Get the network usage statistics
        return self.contract.functions.getNetworkUsage().call()

    def get_user_activity(self, user: str) -> dict:
        # Get the user's activity statistics
        return self.contract.functions.getUserActivity(user).call()

    def get_asset_performance(self, asset: str) -> dict:
        # Get the performance statistics of an asset
        return self.contract.functions.getAssetPerformance(asset).call()

    def get_reward_distribution(self) -> dict:
        # Get the reward distribution statistics
        return self.contract.functions.getRewardDistribution().call()
