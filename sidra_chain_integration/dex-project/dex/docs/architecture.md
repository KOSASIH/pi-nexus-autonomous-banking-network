# Architecture of Sidra Decentralized Exchange

The Sidra Decentralized Exchange is built using a microservices architecture, with the following components:

## Dex App

The Dex App is the core component of the exchange, responsible for managing orders, trades, and liquidity pools. It is built using Solidity and deployed on the Ethereum blockchain.

### Components

* **Order Book**: The order book is responsible for managing buy and sell orders.
* **Trade Engine**: The trade engine is responsible for executing trades and updating the order book.
* **Liquidity Pool**: The liquidity pool is responsible for managing liquidity providers and their corresponding liquidity.

### Interactions

* **User**: Users interact with the Dex App to place orders, execute trades, and provide liquidity.
* **Wallet**: The Wallet component interacts with the Dex App to manage user balances and execute trades.
* **Security**: The Security component interacts with the Dex App to ensure the security and integrity of the exchange.

## Wallet

The Wallet component is responsible for managing user balances and executing trades. It is built using JavaScript and interacts with the Dex App to manage user balances.

### Components

* **Balance Manager**: The balance manager is responsible for managing user balances.
* **Trade Executor**: The trade executor is responsible for executing trades and updating user balances.

### Interactions

* **Dex App**: The Wallet component interacts with the Dex App to manage user balances and execute trades.
* **User**: Users interact with the Wallet component to manage their balances and execute trades.

## Security

The Security component is responsible for ensuring the security and integrity of the exchange. It is built using JavaScript and interacts with the Dex App and Wallet component to ensure the security of the exchange.

### Components

* **Access Control**: The access control component is responsible for managing access to the exchange.
* **Data Encryption**: The data encryption component is responsible for encrypting sensitive data.

### Interactions

* **Dex App**: The Security component interacts with the Dex App to ensure the security of the exchange.
* **Wallet**: The Security component interacts with the Wallet component to ensure the security of user balances and trades.
