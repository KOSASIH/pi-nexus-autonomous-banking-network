# Privacy Protocol

This directory contains the implementation of a zero-knowledge proof smart contract and a Python service for managing privacy features.

## Directory Structure

- `zk_proofs.sol`: Smart contract for creating and verifying zero-knowledge proofs.
- `privacy_service.py`: Python script for interacting with the zero-knowledge proof smart contract.
- `README.md`: Documentation for thePrivacy protocol.

## Zero-Knowledge Proofs (`zk_proofs.sol`)

The ZKProofs contract allows users to create and verify zero-knowledge proofs. It stores proof commitments and associated data securely on the blockchain.

### Functions

- `createProof(bytes32 _proofId, bytes32 _commitment, bytes32 _proofData)`: Allows the owner to create a new zero-knowledge proof.
- `verifyProof(bytes32 _proofId)`: Verifies the existence and validity of a zero-knowledge proof.
- `getProofDetails(bytes32 _proofId)`: Retrieves details of a specific proof.

### Events

- `ProofCreated(bytes32 indexed proofId, bytes32 commitment, address indexed verifier)`: Emitted when a new proof is created.
- `ProofVerified(bytes32 indexed proofId, bool isValid)`: Emitted when a proof is verified.

## Privacy Service (`privacy_service.py`)

The Privacy Service is a Flask application that interacts with the ZKProofs smart contract. It provides endpoints for creating and retrieving zero-knowledge proofs.

### Endpoints

- `POST /create_proof`: Creates a new zero-knowledge proof. Expects a JSON body with `proofId`, `commitment`, and `proofData`.
- `GET /get_proof/<proof_id>`: Retrieves details of a specific proof by its ID.

### Requirements

- Python 3.x
- Flask
- Web3.py

### Installation

1. Install the required packages:
   ```bash
   pip install Flask web3
   ```

2. Update the Ethereum node URL and contract address in the privacy_service.py file.

3. Run the service:
   ```bash
   1 python privacy_service.py
   ```

## License
This project is licensed under the MIT License.
