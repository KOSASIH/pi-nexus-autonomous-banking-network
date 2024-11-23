from flask import Flask, request, jsonify
from web3 import Web3
from risk_assessment_model import RiskAssessmentModel

app = Flask(__name__)

# Initialize the risk assessment model
model = RiskAssessmentModel()
model.load_model('risk_assessment_model.pkl')  # Load pre-trained model

# Connect to Ethereum network
w3 = Web3(Web3.HTTPProvider('https://your.ethereum.node'))

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    data = request.json
    input_data = data['features']  # Expecting a list of features for prediction
    risk_prediction = model.predict(input_data)
    
    # Here you can integrate with blockchain if needed
    # For example, store the prediction on the blockchain

    return jsonify({'risk_prediction': risk_prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
