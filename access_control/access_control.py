import hashlib
import hmac
from flask import request, jsonify

class AccessControl:
    def __init__(self, secret_key):
        self.secret_key = secret_key

    def authenticate(self, username, password):
        # Implement your authentication logic here
        # For demonstration purposes, we'll use a simple username/password combo
        if username == "admin" and password == "password":
            return True
        return False

    def authorize(self, role):
        # Implement your authorization logic here
        # For demonstration purposes, we'll use a simple role-based access control
        if role == "admin":
            return True
        return False

    def generate_token(self, username, role):
        payload = {"username": username, "role": role}
        token = hmac.new(self.secret_key.encode(), json.dumps(payload).encode(), hashlib.sha256).hexdigest()
        return token

    def verify_token(self, token):
        try:
            payload = json.loads(hmac.new(self.secret_key.encode(), token.encode(), hashlib.sha256).hexdigest())
            return payload["username"], payload["role"]
        except Exception:
            return None, None

# Example usage:
app = Flask(__name__)
access_control = AccessControl("my_secret_key")

@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    if access_control.authenticate(username, password):
        token = access_control.generate_token(username, "admin")
        return jsonify({"token": token})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/protected", methods=["GET"])
def protected():
    token = request.headers.get("Authorization")
    username, role = access_control.verify_token(token)
    if username and access_control.authorize(role):
        return jsonify({"message": "Welcome, " + username})
    return jsonify({"error": "Unauthorized"}), 401

if __name__ == "__main__":
    app.run(debug=True)
