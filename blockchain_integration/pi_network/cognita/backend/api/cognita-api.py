from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/data', methods=['GET'])
def get_data():
    # Data retrieval logic
    data = {'message': 'Hello, World!'}
    return jsonify(data)

@app.route('/api/model', methods=['POST'])
def train_model():
    # Model training logic
    model = request.get_json()
    # Train the model
    return jsonify({'message': 'Model trained successfully'})

if __name__ == '__main__':
    app.run(debug=True)
