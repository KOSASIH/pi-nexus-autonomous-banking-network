#!/bin/bash

# Set up the testing environment for Quantum Nexus Protocol

echo "Setting up test environment..."

# Step 1: Install testing dependencies
echo "Installing testing dependencies..."
npm install --save-dev mocha chai

# Step 2: Set up test database
echo "Setting up test database..."
npm run setup-test-db

# Step 3: Run tests
echo "Running tests..."
npm test

echo "Test environment setup completed successfully!"
