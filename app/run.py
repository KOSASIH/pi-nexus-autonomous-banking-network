# run.py

# Import required libraries
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from app.routes import auth_routes, main_routes, transaction_routes, user_routes

# Initialize Flask app
app = Flask(__name__)

# Configure app
app.config.from_object("config.Config")

# Initialize CORS, SQLAlchemy, Migrate, Marshmallow, and JWTManager
CORS(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
ma = Marshmallow(app)
jwt = JWTManager(app)

# Import routes

# Register routes
app.register_blueprint(main_routes.bp)
app.register_blueprint(auth_routes.bp)
app.register_blueprint(user_routes.bp)
app.register_blueprint(transaction_routes.bp)

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
