# __init__.py

from marshmallow import Schema, fields

# Base schema for common fields
class BaseSchema(Schema):
    id = fields.Int(required=True)
    created_at = fields.DateTime(required=True)
    updated_at = fields.DateTime(required=True)

# User schema for user-related data
class UserSchema(BaseSchema):
    username = fields.Str(required=True)
    email = fields.Email(required=True)
    password = fields.Str(required=True)

# Transaction schema for transaction-related data
class TransactionSchema(BaseSchema):
    sender = fields.Str(required=True)
    recipient = fields.Str(required=True)
    amount = fields.Float(required=True)

# Additional schemas can be defined here
class AccountSchema(BaseSchema):
    account_number = fields.Str(required=True)
    balance = fields.Float(required=True)

# Exporting schemas for easy access
__all__ = ['User Schema', 'TransactionSchema', 'AccountSchema']

# Optional: Function to initialize schemas if needed
def init_schemas(app):
    """Initialize schemas if any app-specific configuration is needed."""
    pass
