# PiSure

PiSure is a decentralized insurance platform built using Node.js, PostgreSQL, and MongoDB.

## Getting Started

### Prerequisites

* Docker installed on your system
* Node.js installed on your system (optional)

### Running the Application

1. Clone the repository: `git clone https://github.com/KOSASIH/pi-nexus-autonomous-banking-network/new/main/blockchain_integration/PiSure.git`
2. Change into the repository directory: `cd PiSure`
3. Build and start the application: `docker-compose up`
4. Access the application at: `http://localhost:3000`

### Development

1. Install dependencies: `npm install`
2. Start the application: `npm start`
3. Access the application at: `http://localhost:3000`

### Docker Compose

The `docker-compose.yml` file is used to define the services and their configurations. The services include:

* `app`: The Node.js application
* `db`: The PostgreSQL database
* `mongo`: The MongoDB database

### Dockerfile

The `Dockerfile` is used to build the Docker image for the Node.js application. It installs the dependencies, copies the code, and sets up the environment.

## Contributing

Contributions are welcome! Please submit a pull request with your changes and a brief description of what you've added or fixed.

## License

PiSure is licensed under the MIT License. See [LICENSE](LICENSE) for details.
