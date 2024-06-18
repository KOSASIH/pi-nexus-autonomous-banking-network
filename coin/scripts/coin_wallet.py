from services.wallet_service import WalletService
from models.wallet import Wallet

def create_wallet(user_id: int, coin_id: int) -> Wallet:
    config =...  # load config from config.json
    wallet_service = WalletService(config)
    wallet = Wallet(user_id=user_id, coin_id=coin_id)
    created_wallet = wallet_service.create_wallet(wallet)
    return created_wallet
