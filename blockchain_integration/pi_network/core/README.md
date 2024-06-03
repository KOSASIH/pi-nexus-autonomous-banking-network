# Pi Network

This is the Pi Network implementation for the pi-nexus-autonomous-banking-network.

## Getting Started

To get started, clone the repository and install the dependencies:

```bash
1. git clone https://github.com/KOSASIH/pi-nexus-autonomous-banking-network.git
2. cd pi-nexus-autonomous-banking-network
3. pip install -r blockchain_integration/pi_network/requirements.txt
```

# Configuration

The application can be configured using environment variables:

- PI_NETWORK_PORT: The port on which the Pi Network will listen (default: 8080)
- PI_NETWORK_HOST: The host on which the Pi Network will listen (default: localhost)
- BLOCKCHAIN_NODE_URL: The URL of the blockchain node

# Running the Application

To run the application, execute the following command:

```bash
1. python blockchain_integration/pi_network/core/pi_network.py
```

# API

The Pi Network exposes a RESTful API for interacting with the network.

Users

`GET /users: Retrieve a list of users`

Transactions

`GET /transactions: Retrieve a list of transactions`

# Contributing

Contributions are welcome! To contribute, follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Commit your changes
4. Push your changes to your forked repository
5. Open a pull request

# License

This project is licensed under the MIT License and PIOS.
