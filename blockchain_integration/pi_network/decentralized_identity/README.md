# Decentralized Identity Management

This directory contains the implementation of a decentralized identity management system using Ethereum smart contracts and a Python interface.

## Directory Structure

- `DID.sol`: Smart contract for managing decentralized identities.
- `identity_management.py`: Python script for interacting with the smart contract.
- `README.md`: Documentation for the identity management system.

## Smart Contract (`DID.sol`)

The smart contract allows users to register, update, and retrieve their identities on the blockchain. It emits events for registration and updates.

### Functions

- `registerIdentity(string _name, string _email)`: Registers a new identity.
- `updateIdentity(string _name, string _email)`: Updates an existing identity.
- `getIdentity(address _user)`: Retrieves the identity information for a given user.
- `identityExists(address _user)`: Checks if an identity exists for a given user.

## Python Interface (`identity_management.py`)

The Python script provides an interface to interact with the smart contract.

### Usage

1. Install the required dependencies:
   ```bash
   1 pip install web3
   ```

2. Update the provider_url, contract_address, and abi in the script.

3. Run the script to register or update identities:

   ```bash
   1 python identity_management.py
   ```
### Example
The script includes examples for registering, updating, and retrieving identities. Make sure to replace the placeholders with your actual Ethereum account details and contract information.

## License
This project is licensed under the MIT License and PiOS.
