"""Flask HTTP entrypoint for the DAICTD threat-detection service."""

from flask import Flask, jsonify, request

from ai_engine import AIEngine
from blockchain import Blockchain
from network import Network
from security import (
    SecurityIncidentResponse,
    SecurityOracle,
    SecurityOracleAPI,
)
from threat_detection_model import ThreatDetectionModel
from threat_mitigation_model import ThreatMitigationModel

app = Flask(__name__)

# Initialize threat detection model
threat_detection_model = ThreatDetectionModel()

# Initialize threat mitigation model
threat_mitigation_model = ThreatMitigationModel()

# Initialize AI engine
ai_engine = AIEngine()

# Initialize blockchain
blockchain = Blockchain()

# Initialize network
network = Network()

# Initialize security oracle and incident response
security_oracle = SecurityOracle("private_key.pem", "public_key.pub")
security_incident_response = SecurityIncidentResponse(
    SecurityOracleAPI(security_oracle)
)


@app.route("/detect_threat", methods=["POST"])
def detect_threat():
    """Score incoming traffic with the threat detection model."""
    data = request.json
    threat_score = threat_detection_model.predict(data)
    return jsonify({"threat_score": threat_score})


@app.route("/mitigate_threat", methods=["POST"])
def mitigate_threat():
    """Build a mitigation plan for incoming traffic."""
    data = request.json
    mitigation_plan = threat_mitigation_model.generate_mitigation_plan(data)
    return jsonify({"mitigation_plan": mitigation_plan})


@app.route("/process_input", methods=["POST"])
def process_input():
    """Run incoming input through the AI engine."""
    data = request.json
    output = ai_engine.process_input(data)
    return jsonify({"output": output})


@app.route("/add_block", methods=["POST"])
def add_block():
    """Add an incoming block to the blockchain."""
    data = request.json
    blockchain.add_block(data)
    return jsonify({"block_added": True})


@app.route("/send_data", methods=["POST"])
def send_data():
    """Send incoming data through the network."""
    data = request.json
    network.send_data(data)
    return jsonify({"data_sent": True})


@app.route("/report_incident", methods=["POST"])
def report_incident():
    """Report an incident to the security oracle."""
    data = request.json
    response = security_incident_response.report_incident(data)
    return jsonify(response)


@app.route("/verify_response", methods=["POST"])
def verify_response():
    """Verify an incident response against the security oracle."""
    response = request.json["response"]
    signature = request.json["signature"]
    verified = security_incident_response.verify_response(response, signature)
    return jsonify({"verified": verified})


if __name__ == "__main__":
    app.run(debug=True)
