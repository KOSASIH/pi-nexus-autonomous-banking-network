import requests

def load_encrypted_data_from_pi_network(rpc_url, chain_id):
    # Send request to Pi Network RPC to retrieve encrypted data
    response = requests.post(rpc_url, json={'method': 'get_encrypted_data', 'chain_id': chain_id})

    # Parse response and return encrypted data
    if response.status_code == 200:
        return response.json()['result']
    else:
        raise Exception('Failed to load encrypted data from Pi Network')
