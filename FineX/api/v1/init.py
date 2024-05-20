from flask import Blueprint

# Initialize the blueprint
api_bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# Import the views
from .users import *
from .transactions import *

# Import the schemas
from .schemas.user_schema import *
from .schemas.transaction_schema import *
