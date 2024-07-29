# Pi Network API Reference

Welcome to the Pi Network API Reference! This guide provides detailed documentation on the Pi Network's API, including endpoints, parameters, and response formats.

## Endpoints

### Node API

* `GET /node/info`: Retrieves information about the node, including its version and configuration.
* `GET /node/peers`: Retrieves a list of peers connected to the node.
* `POST /node/send_transaction`: Sends a transaction to the node.

### Smart Contract API

* `GET /contract/{contract_address}`: Retrieves information about a smart contract, including its code and storage.
* `POST /contract/{contract_address}/call`: Calls a function on a smart contract.
* `POST /contract/{contract_address}/deploy`: Deploys a new smart contract.

### Wallet API

* `GET /wallet/{wallet_address}`: Retrieves information about a wallet, including its balance and transaction history.
* `POST /wallet/{wallet_address}/send`: Sends a transaction from a wallet.
* `POST /wallet/{wallet_address}/receive`: Receives a transaction into a wallet.

## Parameters

* `node_id`: The ID of the node to interact with.
* `contract_address`: The address of the smart contract to interact with.
* `wallet_address`: The address of the wallet to interact with.
* `transaction_data`: The data to be sent in a transaction.

## Response Formats

* `JSON`: The default response format for API requests.
* `Error`: An error response format that includes an error message and code.

## Error Codes

* `400`: Bad request.
* `401`: Unauthorized.
* `404`: Not found.
* `500`: Internal server error.

## Next Steps

* Learn more about the Pi Network's architecture and features in our [Getting Started](getting_started.md) guide.
* Explore the Pi Network's smart contract platform in our [Smart Contract Guide](smart_contract_guide.md).
