import os
import sys
from flask import Flask, request, jsonify
from middleware.authentication import authenticate
from middleware.rate_limiting import rate_limit

app = Flask(__name__)

if os.environ.get('FLASK_ENV') == 'production':
    app.config.from_file('config/app.prod.yaml')
else:
    app.config.from_file('config/app.dev.yaml')

@app.route('/')
@authenticate
@rate_limit
def index():
    return jsonify({'message': 'Welcome to the Pi-Nexus Autonomous Banking Network API Gateway'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
