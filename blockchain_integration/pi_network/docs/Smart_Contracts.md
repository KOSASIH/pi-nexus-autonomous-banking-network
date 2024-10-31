# Smart Contracts Documentation

## Overview
This document provides an overview of the smart contracts used in the dApp. The contracts are deployed on the Ethereum blockchain and facilitate various functionalities of the application.

## Contract Name: ExampleContract
### Purpose
The `ExampleContract` is designed to manage user interactions and store data on the blockchain.

### Functions
1. **getDetails**
   - **Description**: Retrieves the name and value stored in the contract.
   - **Returns**: 
     - `string`: Name of the contract.
     - `uint256`: A numeric value associated with the contract.

2. **interact**
   - **Description**: Allows users to interact with the contract by sending a value.
   - **Parameters**:
     - `value` (uint256): The value to be sent to the contract.
   - **Returns**: 
     - `bool`: Indicates whether the interaction was successful.

### Deployment
- **Network**: Ethereum Mainnet
- **Contract Address**: `0xYourContractAddressHere`
- **ABI**: 
```json
1 [
2     "function getDetails() view returns (string memory, uint256)",
3     "function interact(uint256 value) public returns (bool)"
4 ]
```
