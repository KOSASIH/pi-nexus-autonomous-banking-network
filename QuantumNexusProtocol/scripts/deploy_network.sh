#!/bin/bash

# Deploy the Quantum Nexus Protocol network

echo "Starting network deployment..."

# Step 1: Install dependencies
echo "Installing dependencies..."
npm install

# Step 2: Start the blockchain node
echo "Starting blockchain node..."
node src/core/node.js &

# Step 3: Deploy smart contracts
echo "Deploying smart contracts..."
truffle migrate --network development

# Step 4: Start the frontend server
echo "Starting frontend server..."
cd src/frontend
npm start &

echo "Network deployment completed successfully!"
