from marshmallow import Schema, fields


class TransactionSchema(Schema):
    """
    A Transaction schema for the FineX project.
    """

    id = fields.Int(dump_only=True)
    amount = fields.Float(required=True)
    type = fields.Str(required=True)
    timestamp = fields.DateTime(dump_only=True)
    user_id = fields.Int(required=True)
