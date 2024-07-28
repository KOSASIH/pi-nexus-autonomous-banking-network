from flask import Flask, request, jsonify
from data_utils import load_data, preprocess_data
from model_utils import create_model, train_model, evaluate_model
from threat_detection import detect_threats

app = Flask(__name__)

# Load and preprocess data
data = load_data('data/raw/network_traffic_data.csv')
data = preprocess_data(data)

# Create and train the machine learning model
model = create_model(input_shape=(data.shape[1],), num_classes=2)
model = train_model(model, data, epochs=10)

# Evaluate the machine learning model
accuracy, report, matrix = evaluate_model(model, data)
print(f'Accuracy: {accuracy:.3f}')
print(f'Report:\n{report}')
print(f'Matrix:\n{matrix}')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive network traffic data from the client
    data = request.get_json()
    data = np.array(data)

    # Detect threats in real-time
    threats = detect_threats(model, data)

    # Return the threat detection results
    return jsonify({'threats': threats.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
