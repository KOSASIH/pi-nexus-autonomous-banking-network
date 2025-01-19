# Quantum Libraries

This directory contains custom libraries and utilities designed to facilitate quantum computing and quantum machine learning using Qiskit. The libraries provide essential functions for circuit creation, state representation, noise simulation, data preprocessing, and feature mapping.

## Table of Contents

- [Overview](#overview)
- [Libraries](#libraries)
  - [Qiskit Utilities](#qiskit-utilities)
  - [Quantum State Representation](#quantum-state-representation)
  - [Quantum Noise Simulation](#quantum-noise-simulation)
  - [Data Preprocessing for Quantum Machine Learning](#data-preprocessing-for-quantum-machine-learning)
  - [Feature Maps for Quantum Machine Learning](#feature-maps-for-quantum-machine-learning)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to provide a set of libraries that simplify the development of quantum algorithms and applications. Each library is designed to address specific aspects of quantum computing, making it easier for researchers and developers to implement and experiment with quantum algorithms.

## Libraries

### Qiskit Utilities
- **File**: `qiskit_utils.py`
- **Description**: Contains utility functions for creating and manipulating quantum circuits, running simulations, and visualizing results. Functions include circuit creation, state preparation, and noise application.

### Quantum State Representation
- **File**: `quantum_state.py`
- **Description**: Provides a class for representing quantum states, including methods for applying gates, measuring states, and visualizing results. This class encapsulates functionalities related to quantum state manipulation.

### Quantum Noise Simulation
- **File**: `quantum_noise.py`
- **Description**: Implements functions for simulating noise in quantum systems. It includes the creation of noise models (e.g., depolarizing, amplitude damping) and applying noise to quantum circuits, allowing for realistic simulations of quantum operations.

### Data Preprocessing for Quantum Machine Learning
- **File**: `quantum_data_preprocessing.py`
- **Description**: Contains functions for preprocessing data for quantum machine learning applications. This includes normalization, encoding classical data into quantum states, and splitting datasets for training and testing.

### Feature Maps for Quantum Machine Learning
- **File**: `quantum_feature_maps.py`
- **Description**: Implements various feature maps used in quantum machine learning, such as Z-Feature Maps and Pauli Feature Maps. These feature maps encode classical data into quantum states, which is essential for quantum algorithms.

## Requirements

- Python 3.6 or higher
- Qiskit
- NumPy
- scikit-learn

## Installation

To install the required packages, you can use pip:

```bash
pip install qiskit numpy scikit-learn
```

## Usage
To use any of the libraries, you can import the relevant module in your Python script. For example, to use the Qiskit utilities:

```python
1 from libraries.qiskit_utils import create_hadamard_circuit, run_circuit
```
Make sure to adjust the parameters as needed for each function.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features to implement, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
