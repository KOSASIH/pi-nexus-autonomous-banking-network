from marshmallow import fields, Schema

class UserSchema(Schema):
    """
    A User schema for the FineX project.
    """
    id = fields.Int(dump_only=True)
    username = fields.Str(required=True)
    email = fields.Str(required=True)
    password = fields.Str(load_only=True)
