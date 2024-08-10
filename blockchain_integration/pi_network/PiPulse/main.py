from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from config import config

app = Flask(__name__)
app.config.from_object(config)

# Initialize database
db = SQLAlchemy(app)

# Initialize JWT manager
jwt = JWTManager(app)

# Define routes
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    # Return metrics data from database
    metrics = db.session.query(Metric).all()
    return jsonify([metric.to_dict() for metric in metrics])

@app.route('/api/login', methods=['POST'])
def login():
    # Authenticate user and return JWT token
    username = request.json.get('username')
    password = request.json.get('password')
    user = db.session.query(User).filter_by(username=username).first()
    if user and user.check_password(password):
        token = jwt.create_access_token(identity=username)
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

if __name__ == '__main__':
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)
