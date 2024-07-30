from .views import coin_view

def create_coin(name, symbol, value):
    coin = Coin(name=name, symbol=symbol, value=value)
    db.session.add(coin)
    db.session.commit()
    return coin

def update_coin(coin_id, name, symbol, value):
    coin = Coin.query.get(coin_id)
    if coin:
        coin.name = name
        coin.symbol = symbol
        coin.value = value
        db.session.commit()
        return coin
    return None
