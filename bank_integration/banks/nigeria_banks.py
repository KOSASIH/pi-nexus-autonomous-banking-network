import requests


class NigeriaBanks:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_bank_list(self):
        url = f"{self.base_url}/banks"
        response = requests.get(url)
        return response.json()

    def get_bank_details(self, bank_code):
        url = f"{self.base_url}/banks/{bank_code}"
        response = requests.get(url)
        return response.json()

    def get_bank_branches(self, bank_code):
        url = f"{self.base_url}/banks/{bank_code}/branches"
        response = requests.get(url)
        return response.json()
