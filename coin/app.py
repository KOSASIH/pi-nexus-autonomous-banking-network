from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config.app_config import AppConfig

def create_app():
    app = Flask(__name__)
    app.config.from_object(AppConfig)
    db = SQLAlchemy(app)

    # Register blueprints, models, and services here

    return app
