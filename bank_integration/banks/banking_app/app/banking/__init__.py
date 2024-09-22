from flask import Blueprint

banking_blueprint = Blueprint("banking", __name__)

from . import banking, models, schemas
