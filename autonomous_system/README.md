# Autonomous Banking Network

This project is an autonomous banking network that integrates various banks and provides a web application for making predictions based on financial data.

## Requirements

- Python 3.8
- Docker

## Installation

1. Clone the repository:

`1. git clone https://github.com/KOSASIH/autonomous-banking-network.git`

2. Build the Docker image:

```
1. cd autonomous-banking-network
2. docker build -t autonomous-banking-network .
```

3. Run the Docker container:

```
1. docker run -p 5000:5000 autonomous-banking-network
```

Access the web application at http://localhost:5000/predict.

# Usage

The web application provides a single endpoint /predict that accepts POST requests with JSON data containing the three features feature1, feature2, and feature3. The endpoint returns a JSON response containing the predicted target value.

# Testing

To run the tests, use the following command:

```
1. pytest
```

# License

This project is licensed under the MIT License.

# Acknowledgements

This project was created using the following libraries:

1. NumPy
2. Pandas
3. Scikit-learn
4. Flask
5. Pytest

Happy coding .. ☕
