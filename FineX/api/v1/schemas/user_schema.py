# user_schema.py

from marshmallow import Schema, fields, validate

# User Schema
class UserSchema(Schema):
    id = fields.Int(required=True, description="Unique identifier for the user")
    username = fields.Str(required=True, validate=validate.Length(min=3, max=100), description="The username chosen by the user")
    email = fields.Email(required=True, validate=validate.Length(max=100), description="The user's email address")
    password = fields.Str(required=True, validate=validate.Length(min=8, max=200), description="The user's password (hashed for security)")
    first_name = fields.Str(validate=validate.Length(max=50), description="The user's first name")
    last_name = fields.Str(validate=validate.Length(max=50), description="The user's last name")
    phone_number = fields.Str(validate=validate.Length(max=20), description="The user's phone number")
    address = fields.Str(validate=validate.Length(max=200), description="The user's address")
    role = fields.Str(required=True, validate=validate.OneOf(["admin", "user"]), description="The user's role in the system")

    # Additional fields can be added as needed

# Example of how to use the schema
if __name__ == "__main__":
    # Example user data
    user_data = {
        "id": 1,
        "username": "johnDoe",
        "email": "johndoe@example.com",
        "password": "password123",  # Note: In a real application, passwords should be hashed before storing
        "first_name": "John",
        "last_name": "Doe",
        "phone_number": "+1234567890",
        "address": "123 Main St",
        "role": "user"
    }

    # Create an instance of the schema
    schema = UserSchema()

    # Validate and serialize the user data
    result = schema.load(user_data)
    print("Validated User Data:", result)

    # Serialize the user data
    serialized_data = schema.dump(result)
    print("Serialized User Data:", serialized_data)
