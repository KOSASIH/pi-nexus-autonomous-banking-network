from flask import Flask, request, jsonify
from device_auth import DeviceAuth
from iot_devices import SmartHomeDevice, WearableDevice, AutonomousVehicleDevice
from microtransactions import MicrotransactionHandler
from data_encryption import encrypt_data

app = Flask(__name__)

# Initialize the device authentication instance
device_auth = DeviceAuth('device_id', 'device_secret')

# Initialize the microtransaction handler instance
microtransaction_handler = MicrotransactionHandler('microtransactions.db')

@app.route('/iot/devices', methods=['GET'])
def get_iot_devices():
    # Return a list of supported IoT devices
    return jsonify(['Smart Home', 'Wearable', 'Autonomous Vehicle'])

@app.route('/iot/devices/<device_type>', methods=['POST'])
def register_iot_device(device_type):
    # Register a new IoT device
    device_id = request.json['device_id']
    device_token = request.json['device_token']
    if device_type == 'Smart Home':
        device = SmartHomeDevice(device_id, device_token)
    elif device_type == 'Wearable':
        device = WearableDevice(device_id, device_token)
    elif device_type == 'Autonomous Vehicle':
        device = AutonomousVehicleDevice(device_id, device_token)
    else:
        return jsonify({'error': 'Invalid device type'}), 400
    return jsonify({'device_id': device_id, 'device_token': device_token})

@app.route('/iot/transactions', methods=['POST'])
def make_iot_transaction():
    # Make a transaction using an IoT device
    device_id = request.json['device_id']
    transaction_id = request.json['transaction_id']
    amount = request.json['amount']
    device = get_iot_device(device_id)
    if device:
        # Authenticate the device
        if device_auth.authenticate(request):
            # Make the transaction
            if device.make_payment(amount):
                # Create a microtransaction
                microtransaction_handler.create_microtransaction(transaction_id, amount)
                return jsonify({'transaction_id': transaction_id, 'amount': amount})
            else:
                return jsonify({'error': 'Transaction failed'}), 400
        else:
            return jsonify({'error': 'Device authentication failed'}), 401
    else:
        return jsonify({'error': 'Invalid device ID'}), 404

def get_iot_device(device_id):
    # Retrieve an IoT device instance
    if device_id.startswith('SH'):
        return SmartHomeDevice(device_id, 'device_token')
    elif device_id.startswith('WR'):
        return WearableDevice(device_id, 'device_token')
    elif device_id.startswith('AV'):
        return AutonomousVehicleDevice(device_id, 'device_token')
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)
