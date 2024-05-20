import os
from logging.config import dictConfig
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_marshmallow import Marshmallow
from flask_restplus import Api, Resource

# Initialize app
app = Flask(__name__)

# Configure app
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'formatter': 'default',
        'stream': 'ext://flask.logging.wsgi_errors_stream'
    }},
    'root': {
        'handlers': ['wsgi'],
        'level': 'INFO'
    }
})

# Set up configurations
app_settings = os.getenv('APP_SETTINGS')
app.config.from_object(app_settings)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
ma = Marshmallow(app)
api = Api(app)
CORS(app)

# Import blueprints
from api.v1.authentication import authentication_blueprint
from api.v1.users import users_blueprint
from api.v1.transactions import transactions_blueprint

# Register blueprints
app.register_blueprint(authentication_blueprint, url_prefix='/api/v1/auth')
app.register_blueprint(users_blueprint, url_prefix='/api/v1/users')
app.register_blueprint(transactions_blueprint, url_prefix='/api/v1/transactions')

# Start the application
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])
