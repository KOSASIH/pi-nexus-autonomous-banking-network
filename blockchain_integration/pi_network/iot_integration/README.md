# IoT Payment Service

This IoT Payment Service allows IoT devices to make payments using Ethereum smart contracts. The service provides endpoints to process payments and manage IoT devices.

## Requirements

- Python 3.x
- Flask
- Web3.py
- JSON

## Installation

1. **Clone the repository**:
   ```bash
   1 git clone https://github.com/KOSASIH/pi-nexus-autonomous-banking-network.git
   2 cd iot-payment-service
   ```

2. **Install the required Python packages**:

   ```bash
   1 pip install Flask web3
   ```

3. Set up environment variables: Create a .env file in the root directory and add the following variables:

   ```plaintext
   1 ETH_NODE_URL=https://your.ethereum.node
   2 CONTRACT_ADDRESS=YOUR_CONTRACT_ADDRESS
   3 PRIVATE_KEY=YOUR_PRIVATE_KEY

4. **Create a JSON file for the smart contract ABI**: Save your smart contract ABI in a file named contract_abi.json in the same directory as iot_payment_service.py.

### Running the Service

1. **Run the Flask application**:

   ```bash
   1 python iot_payment_service.py
   ```
2. **Open your web browser** and navigate to http://127.0.0.1:5000/api/iot/devices to view the list of IoT devices.

### API Endpoints

1. **Get IoT Devices**
- **Endpoint**: /api/iot/devices
- **Method**: GET
- **Response**: Returns a list of registered IoT devices and their owners.

2. **Make Payment**
- **Endpoint**: /api/iot/payment
- **Method**: POST
