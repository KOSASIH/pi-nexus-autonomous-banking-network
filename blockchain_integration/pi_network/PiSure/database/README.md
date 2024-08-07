# PiSure Database

This repository contains the database configuration and models for PiSure, a decentralized insurance platform.

## Database Configuration

The database configuration is split into two directories: `postgres` and `mongodb`. The `postgres` directory contains the configuration for the PostgreSQL database, while the `mongodb` directory contains the configuration for the MongoDB database.

## Migrations

The `migrations` directory contains the database migrations using TypeORM. To run the migrations, use the following command:

`npm run migrate`

To revert the migrations, use the following command:

`npm run revert`

## Models

The `models` directory contains the database models using TypeORM. The models are used to interact with the database.

## Contributing

Contributions are welcome! Please submit a pull request with your changes and a brief description of what you've added or fixed.
