# Development Guide for Sidra Decentralized Exchange

This guide provides an overview of the development process for the Sidra Decentralized Exchange.

## Setting up the Development Environment

1. Clone the exchange's repository: `git clone https://github.com/KOSASIH/pi-nexus-autonomous-banking-network.git`
2. Navigate to the `dex-project/dex` directory: `cd pi-nexus-autonomous-banking-network/sidra_chain_integration/dex-project/dex`
3. Install the dependencies: `npm install`
4. Start the development environment: `npm run dev`

## Building the Exchange

1. Build the Dex App: `npm run build:dex-app`
2. Build the Wallet: `npm run build:wallet`
3. Build the Security component: `npm run build:security`

## Testing the Exchange

1. Run the unit tests: `npm run test:unit`
2. Run the integration tests: `npm run test:integration`

## Deploying the Exchange

1. Deploy the Dex App: `npm run deploy:dex-app`
2. Deploy the Wallet: `npm run deploy:wallet`
3. Deploy the Security component: `npm run deploy:security`

## Contributing to the Exchange

If you would like to contribute to the Sidra Decentralized Exchange, please follow these steps:

1. Fork the exchange's repository: `git fork https://github.com/KOSASIH/pi-nexus-autonomous-banking-network.git`
2. Create a new branch: `git branch my-feature`
3. Make your changes: `git add .` and `git commit -m "My feature"`
4. Push your changes: `git push origin my-feature`
5. Create a pull request: `git request-pull origin my-feature`
