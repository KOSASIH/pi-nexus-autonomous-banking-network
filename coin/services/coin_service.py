from models.coin import Coin
from utils.security import encrypt, decrypt

class CoinService:
    def __init__(self, config: dict):
        self.config = config

    def mint_coin(self, coin: Coin) -> Coin:
        # Minting logic here
        encrypted_coin = encrypt(coin, self.config["encryption_key"])
        return encrypted_coin

    def get_coin(self, coin_id: int) -> Optional[Coin]:
        # Database query logic here
        coin_data =...  # retrieve coin data from database
        decrypted_coin = decrypt(coin_data, self.config["decryption_key"])
        return decrypted_coin
