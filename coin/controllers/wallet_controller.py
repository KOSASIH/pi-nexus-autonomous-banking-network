from.views import wallet_view

def create_wallet(user_id, balance):
    wallet = Wallet(user_id=user_id, balance=balance)
    db.session.add(wallet)
    db.session.commit()
    return wallet

def update_wallet(wallet_id, balance):
    wallet = Wallet.query.get(wallet_id)
    if wallet:
        wallet.balance = balance
        db.session.commit()
        return wallet
    return None
