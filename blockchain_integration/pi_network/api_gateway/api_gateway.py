from flask import Flask, request, jsonify
from api_routes import api_routes

app = Flask(__name__)

# Initialize API routes
api_routes(app)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
