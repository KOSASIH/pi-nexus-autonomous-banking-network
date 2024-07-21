import requests


class SidraChain:
    def __init__(self, api_key, api_secret, network):
        self.api_key = api_key
        self.api_secret = api_secret
        self.network = network

    def create_account(self):
        # Create a new account on the Sidra Chain network
        response = requests.post(
            f"https://{self.network}.sidra.chain/api/v1/accounts",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return response.json()

    def create_smart_contract(self, contract_name, functions):
        # Create a new smart contract on the Sidra Chain network
        response = requests.post(
            f"https://{self.network}.sidra.chain/api/v1/contracts",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"name": contract_name, "functions": functions},
        )
        return response.json()

    def deploy_smart_contract(self, contract_name):
        # Deploy the smart contract on the Sidra Chain network
        response = requests.post(
            f"https://{self.network}.sidra.chain/api/v1/contracts/{contract_name}/deploy",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return response.json()
