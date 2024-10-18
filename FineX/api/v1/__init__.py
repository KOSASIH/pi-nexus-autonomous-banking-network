# __init__.py

from flask import Blueprint

# Create a blueprint for the API version 1
api_v1 = Blueprint('api_v1', __name__)

# Import routes for the API
from . import routes

# Optional: Add any initialization code here
def init_app(app):
    """Initialize the API with the given Flask app."""
    app.register_blueprint(api_v1, url_prefix='/api/v1')

# Optional: Add any error handlers or middleware here
@api_v1.errorhandler(404)
def not_found(error):
    return {"message": "Resource not found"}, 404

@api_v1.errorhandler(500)
def internal_error(error):
    return {"message": "Internal server error"}, 500
