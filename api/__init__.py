from flask import Flask
from .endpoints import endpoints
from .authentication import jwt_manager

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.register_blueprint(endpoints, url_prefix='/api')
    jwt_manager.init_app(app)
    return app
