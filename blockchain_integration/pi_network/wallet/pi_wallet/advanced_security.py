import hashlib
import hmac
import secrets
import time
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///security.db"
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    two_factor_secret = db.Column(db.String(32), nullable=True)
    multi_factor_secret = db.Column(db.String(32), nullable=True)
    transaction_limit = db.Column(db.Integer, nullable=False, default=1000)
    ip_whitelist = db.Column(db.String(128), nullable=True)
    ip_blacklist = db.Column(db.String(128), nullable=True)

    def set_password(self, password):
        self.password = hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password):
        return hmac.compare_digest(self.password, hashlib.sha256(password.encode()).hexdigest())

    def generate_two_factor_secret(self):
        self.two_factor_secret = secrets.token_urlsafe(16)

    def generate_multi_factor_secret(self):
        self.multi_factor_secret = secrets.token_urlsafe(16)

    def check_two_factor_code(self, code):
        return hmac.compare_digest(self.two_factor_secret, code)

    def check_multi_factor_code(self, code):
        return hmac.compare_digest(self.multi_factor_secret, code)

@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        # 2FA and MFA checks
        if user.two_factor_secret:
            two_factor_code = request.json["two_factor_code"]
            if not user.check_two_factor_code(two_factor_code):
                return jsonify({"error": "Invalid 2FA code"}), 401
        if user.multi_factor_secret:
            multi_factor_code = request.json["multi_factor_code"]
            if not user.check_multi_factor_code(multi_factor_code):
                return jsonify({"error": "Invalid MFA code"}), 401
        # IP whitelisting/blacklisting
        ip_address = request.remote_addr
        if user.ip_blacklist and ip_address in user.ip_blacklist:
            return jsonify({"error": "IP address is blacklisted"}), 401
        if user.ip_whitelist and ip_address not in user.ip_whitelist:
            return jsonify({"error": "IP address is not whitelisted"}), 401
        # Transaction limit check
        if user.transaction_limit <= 0:
            return jsonify({"error": "Transaction limit exceeded"}), 401
        user.transaction_limit -= 1
        db.session.commit()
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

if __name__ == "__main__":
    app.run(debug=True)
