# Smart Contracts for dApps

## Overview
This directory contains the smart contracts for the dApps Builder project. The contracts are written in Solidity and are designed to facilitate the core functionalities of the decentralized application.

## Contracts
- **DAppContract.sol**: The main contract that handles user deposits, withdrawals, and balance management.
- **Migrations.sol**: A contract to manage the deployment of other contracts.

## Migrations
The migration scripts are used to deploy the contracts to the blockchain. The initial migration script is included in this directory.

## Testing
The test directory contains tests for the DAppContract to ensure that all functionalities work as expected. Tests are written using Mocha and Chai.

## Truffle Configuration
The `truffle-config.js` file contains the configuration for deploying the contracts to different networks, including development, Ropsten, and Mainnet.

## Getting Started

### Prerequisites
- Node.js (v14 or higher)
- Truffle (v5.0 or higher)
- Ganache (for local development)

### Installation
1. Install Truffle globally:
   ```bash
   1 npm install -g truffle
   ```

2. Install dependencies:
   ```bash
   1 npm install
   ```

### Running Migrations
To deploy the contracts to the local development network, run:

   ```bash
   1 truffle migrate --network development
   ```

### Running Tests
To execute the tests, use:

   ```bash
   1 truffle test
   ```

### Author

KOSASIH
