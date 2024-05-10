#!/bin/bash

# Set environment variables
export PI_NEXUS_AUTONOMOUS_BANKING_NETWORK_ENV=production
export DATABASE_URL="postgres://user:password@host:port/dbname"

# Build Docker image
docker build -t pi-nexus-autonomous-banking-network .

# Push image to Docker Hub
docker tag pi-nexus-autonomous-banking-network:latest $DOCKER_HUB_USERNAME/pi-nexus-autonomous-banking-network:latest
docker push $DOCKER_HUB_USERNAME/pi-nexus-autonomous-banking-network:latest

# Deploy to Kubernetes
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/service.yaml

# Verify deployment
kubectl rollout status deployment/pi-nexus-autonomous-banking-network
