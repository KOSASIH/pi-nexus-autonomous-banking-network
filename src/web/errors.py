from flask import Blueprint, render_template, abort, request

errors = Blueprint('errors', __name__)

@errors.app_errorhandler(404)
def handle_404_error(error):
    """Handle a 404 Not Found error."""
    return render_template('errors/404.html'), 404

@errors.app_errorhandler(500)
def handle_500_error(error):
    """Handle a 500 Internal Server Error."""
    return render_template('errors/500.html'), 500

@errors.app_errorhandler(401)
def handle_unauthorized_error(error):
    """Handle a 401 Unauthorized error."""
    return render_template('errors/401.html'), 401

@errors.app_errorhandler(403)
def handle_forbidden_error(error):
    """Handle a 403 Forbidden error."""
    return render_template('errors/403.html'), 403

@errors.app_errorhandler(400)
def handle_bad_request_error(error):
    """Handle a 400 Bad Request error."""
    if request.is_xhr:
        return render_template('errors/400.html'), 400
    else:
        return abort(400)
