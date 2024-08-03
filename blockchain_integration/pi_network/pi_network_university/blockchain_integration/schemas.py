# blockchain_integration/schemas.py
from. import ma
from.models import User, Wallet, Transaction

class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = User
        load_instance = True
        exclude = ("password",)

class WalletSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Wallet
        load_instance = True
        exclude = ("private_key",)

class TransactionSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Transaction
        load_instance = True
