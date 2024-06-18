# exception_middleware.py
from flask import request, jsonify

class ExceptionMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        try:
            return self.app(environ, start_response)
        except Exception as e:
            error_code = getattr(e, "code", 500)
            error_message = str(e)
            response = jsonify({"error": {"code": error_code, "message": error_message}})
            response.status_code = error_code
            return response(environ, start_response)
