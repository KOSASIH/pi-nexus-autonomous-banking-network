# src/web/app.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db = SQLAlchemy(app)

    from src.core.models import BankAccount

    db.create_all()

    from src.web.views import views

    app.register_blueprint(views)

    return app
