# Autonomous Banking Network AI Services

This repository contains the codebase for an autonomous banking network AI system,
adopting a microservices architecture.

## Directory Structure

- `data_ingestion`: Service for data ingestion
- `model_training`: Service for model training
- `prediction`: Service for model prediction
- `docker-compose.yml`: Docker Compose configuration

## Getting Started

1. Install Docker: https://docs.docker.com/get-docker/
2. Clone this repository
3. Run `docker-compose up`

## Services

- `data_ingestion`: Ingests data from various sources and stores it in a database
- `model_training`: Trains AI models using the ingested data
- `prediction`: Provides predictions based on the trained models
