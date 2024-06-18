import sys
from stellar_integration.stellar_config import load_config
from stellar_integration.stellar_wallet import StellarWallet
from stellar_integration.stellar_transaction_builder import StellarTransactionBuilder
from stellar_integration.stellar_transaction_signer import StellarTransactionSigner
from stellar_integration.stellar_transaction_sender import StellarTransactionSender

def main():
    config_file = sys.argv[1]
    config = load_config(config_file)

    wallet = StellarWallet(
        secret_key=config['wallet_secret_key'],
        public_key=config['wallet_public_key']
    )

    transaction_builder = StellarTransactionBuilder(
        source_account=wallet.get_account(),
        destination_account="GCEZWKCA5VLDNRLN3RPRJMRZOX3Z6G5CHVG2NIX",
        amount=100,
        asset_code="USDC",
        asset_issuer="GD62Y3YO3BZ525X5X225X5X5X5X5X5X5X5X5X5X5"
    )

    transaction = transaction_builder.build_transaction()
    signed_transaction = StellarTransactionSigner(wallet).sign_transaction(transaction)

    horizon_url = "https://horizon-testnet.stellar.org"
    transaction_sender = StellarTransactionSender(horizon_url)
    response = transaction_sender.send_transaction(signed_transaction)
    print(response)

if __name__ == "__main__":
    main()
