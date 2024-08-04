#!/bin/bash

# Deploy the blockchain contracts
truffle migrate --network mainnet

# Deploy the AI platform
python ai-platform/src/ai-core/deploy.py

# Deploy the frontend
npm run build
npm run deploy

# Deploy the backend
python backend/api/deploy.py
