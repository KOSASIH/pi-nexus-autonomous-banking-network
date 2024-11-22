# Cross-Chain Bridge

This directory contains the implementation of a smart contract for cross-chain transactions and a Python service for managing cross-chain interactions.

## Directory Structure

- `cross_chain_bridge.sol`: Smart contract for managing cross-chain transactions.
- `bridge_service.py`: Python service for interacting with the cross-chain bridge.
- `README.md`: Documentation for the cross-chain bridge.

## Cross Chain Bridge Contract (`cross_chain_bridge.sol`)

The Cross Chain Bridge contract allows users to initiate and complete cross-chain transfers. It tracks transfer details and emits events for initiated and completed transfers.

### Functions

- `initiateTransfer(uint256 _amount, string memory _targetChain, address _targetAddress)`: Initiates a cross-chain transfer.
- `completeTransfer(bytes32 _transferId)`: Completes a cross-chain transfer.
- `getTransferDetails(bytes32 _transferId)`: Retrieves the details of a specific transfer.

### Events

- `TransferInitiated(address indexed sender, uint256 amount, string targetChain, address targetAddress)`: Emitted when a transfer is initiated.
- `TransferCompleted(address indexed receiver, uint256 amount, string sourceChain)`: Emitted when a transfer is completed.

## Bridge Service (`bridge_service.py`)

The Bridge Service is a Python application that interacts with the Cross Chain Bridge contract. It allows users to initiate transfers, complete transfers, and retrieve transfer details.

### Installation

1. Install the required packages:
   ```bash
   1 pip install web3
   ```

2. Usage

1. **Initialize the Bridge Service**: Update the provider_url, contract_address, and abi in the bridge_service.py file.

2. **Initiate a Transfer**: Call the initiate_transfer method with the sender's account, private key, amount, target chain, and target address.

3. **Complete a Transfer**: Call the complete_transfer method with the transfer ID, sender's account, and private key.

4. **Get Transfer Details**: Call the get_transfer_details method with the transfer ID to retrieve its details.

## License
This project is licensed under the MIT License.

