# transaction_schema.py

from marshmallow import Schema, fields, validate

# Transaction Schema
class TransactionSchema(Schema):
    id = fields.Int(required=True, description="Unique identifier for the transaction")
    sender = fields.Str(required=True, validate=validate.Length(min=1), description="The sender's identifier")
    recipient = fields.Str(required=True, validate=validate.Length(min=1), description="The recipient's identifier")
    amount = fields.Float(required=True, validate=validate.Range(min=0), description="The amount of the transaction")
    timestamp = fields.DateTime(required=True, description="The timestamp of the transaction")
    status = fields.Str(required=True, validate=validate.OneOf(["pending", "completed", "failed"]), description="The status of the transaction")
    currency = fields.Str(required=True, validate=validate.Length(equal=3), description="The currency code (ISO 4217)")

# Example of how to use the schema
if __name__ == "__main__":
    # Example transaction data
    transaction_data = {
        "id": 1,
        "sender": "user123",
        "recipient": "user456",
        "amount": 100.50,
        "timestamp": "2024-10-18T12:00:00",
        "status": "completed",
        "currency": "USD"
    }

    # Create an instance of the schema
    schema = TransactionSchema()

    # Validate and serialize the transaction data
    result = schema.load(transaction_data)
    print("Validated Transaction Data:", result)

    # Serialize the transaction data
    serialized_data = schema.dump(result)
    print("Serialized Transaction Data:", serialized_data)
