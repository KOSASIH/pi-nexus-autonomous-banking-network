# network_utils.py

import requests

def send_request(url, method, data=None, headers=None):
    response = requests.request(method, url, data=data, headers=headers)
    response.raise_for_status()
    return response.json()

def download_file(url, path):
    response = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)
