import json
    import os
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from web3 import Web3
    
    def secure_send_transaction(web3_provider, private_key_hex, to_address, value_wei):
        """
        Securely sign and send a transaction with dynamic gas estimation and nonce tracking.
        """
        account = web3_provider.eth.account.from_key(private_key_hex)
        nonce = web3_provider.eth.get_transaction_count(account.address)
        
        gas_estimate = web3_provider.eth.estimate_gas({
            'from': account.address,
            'to': to_address,
            'value': value_wei
        })
        
        tx = {
            'chainId': web3_provider.eth.chain_id,
            'from': account.address,
            'to': to_address,
            'value': value_wei,
            'gas': gas_estimate,
            'maxFeePerGas': web3_provider.to_wei('50', 'gwei'),
            'maxPriorityFeePerGas': web3_provider.to_wei('2', 'gwei'),
            'nonce': nonce,
            'type': '0x2'
        }
        
        signed_tx = web3_provider.eth.account.sign_transaction(tx, private_key_hex)
        tx_hash = web3_provider.eth.send_raw_transaction(signed_tx.rawTransaction)
        return web3_provider.to_hex(tx_hash)
    
    def secure_generate_keypair():
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return private_pem, public_pem
    