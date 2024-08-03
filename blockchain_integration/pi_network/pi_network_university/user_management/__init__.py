# user_management/__init__.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///user_management.db"
app.config["JWT_SECRET_KEY"] = "super-secret-key"
app.config["BCRYPT_LOG_ROUNDS"] = 12

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

from .models import User
from .schemas import UserSchema
from .routes import user_routes, auth_routes

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
