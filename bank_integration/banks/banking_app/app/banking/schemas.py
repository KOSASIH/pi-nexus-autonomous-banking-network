from marshmallow import Schema, fields

class AccountSchema(Schema):
    id = fields.Int(dump_only=True)
    account_number = fields.Str(required=True)
    balance = fields.Float(required=True)

    class Meta:
        fields = ("id", "account_number", "balance")
