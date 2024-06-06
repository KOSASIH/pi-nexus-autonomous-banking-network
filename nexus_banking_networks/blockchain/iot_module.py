import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# IoT Device Simulator
iot_devices = [
    {"id": 1, "name": "Smart Meter", "data": {"energy_consumption": 100}},
    {"id": 2, "name": "Security Camera", "data": {"motion_detected": True}},
    {"id": 3, "name": "Environmental Sensor", "data": {"temperature": 25, "humidity": 60}}
]

# Blockchain API
blockchain_api = "http://localhost:5000"

@app.route('/iot_data', methods=['POST'])
def receive_iot_data():
    data = request.get_json()
    device_id = data["device_id"]
    device_data = data["device_data"]
    
    # Validate IoT device data
    if device_id in [device["id"] for device in iot_devices]:
        # Send data to blockchain API
        response = requests.post(blockchain_api + '/add_transaction', json={"device_id": device_id, "device_data": device_data})
        if response.status_code == 200:
            return jsonify({"message": "IoT data successfully added to blockchain"}), 200
        else:
            return jsonify({"message": "Error adding IoT data to blockchain"}), 500
    else:
        return jsonify({"message": "Invalid IoT device ID"}), 400

@app.route('/get_iot_data', methods=['GET'])
def get_iot_data():
    device_id = request.args.get('device_id')
    if device_id:
        for device in iot_devices:
            if device["id"] == int(device_id):
                return jsonify(device["data"]), 200
        return jsonify({"message": "IoT device not found"}), 404
    else:
        return jsonify([device["data"] for device in iot_devices]), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
