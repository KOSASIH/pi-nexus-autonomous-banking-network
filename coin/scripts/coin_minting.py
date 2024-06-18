from services.coin_service import CoinService
from models.coin import Coin

def mint_coin(coin_name: str, coin_symbol: str, amount: float) -> Coin:
    config =...  # load config from config.json
    coin_service = CoinService(config)
    coin = Coin(name=coin_name, symbol=coin_symbol, amount=amount)
    minted_coin = coin_service.mint_coin(coin)
    return minted_coin
