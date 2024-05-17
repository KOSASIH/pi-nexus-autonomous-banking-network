# Fantasy Exchange

A simple exchange for the Pi-Nexus Autonomous Banking Network.

## Getting Started

To get started, clone this repository and install the required dependencies:

```bash

1. git clone https://github.com/KOSASIH/pi-nexus-autonomous-banking-network.git
2. cd exchanges/exchange/fantasy
3. pip install -r requirements.txt
```

Next, configure the exchange by setting the following environment variables:

FANTASY_EXCHANGE_API_KEY: The API key for the Fantasy Exchange.
FANTASY_EXCHANGE_API_SECRET: The API secret for the Fantasy Exchange.
DATABASE_URL: The URL for the database.
You can set these environment variables in a .env file:

```mmakefile

1 .env
2. FANTASY_EXCHANGE_API_KEY=your_api_key
3. FANTASY_EXCHANGE_API_SECRET=your_api_secret
4. DATABASE_URL=your_database_url
```

Finally, run the exchange:

```bash

python main.py
```

# Services

The exchange provides the following services:

1. AccountService: Manages accounts.
2. TransactionService: Manages transactions.
3. UserService: Manages users.

# Testing

To run the tests, use the following command:

```bash

1. python -m unittest
```

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
