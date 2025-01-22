# Quantum Benchmarks

This directory contains benchmarking scripts designed to evaluate the performance of various quantum algorithms. Each script measures execution time, solution quality, and other relevant metrics to provide insights into the efficiency and effectiveness of the algorithms.

## Table of Contents

- [Benchmarking Grover's Algorithm](#benchmarking-grovers-algorithm)
- [Benchmarking Shor's Algorithm](#benchmarking-shors-algorithm)
- [Benchmarking Variational Quantum Eigensolver (VQE)](#benchmarking-variational-quantum-eigensolver-vqe)
- [Benchmarking Hybrid Quantum-Classical Algorithms](#benchmarking-hybrid-quantum-classical-algorithms)

## Benchmark Scripts

### Benchmarking Grover's Algorithm
- **File**: `benchmark_grover.py`
- **Description**: This script benchmarks Grover's algorithm by measuring execution time and success probability for different numbers of qubits and marked elements. It visualizes the results to analyze performance trends.

### Benchmarking Shor's Algorithm
- **File**: `benchmark_shor.py`
- **Description**: This script benchmarks Shor's algorithm by measuring execution time and the factors found for different integers. It provides insights into the performance of the algorithm for factoring integers.

### Benchmarking Variational Quantum Eigensolver (VQE)
- **File**: `benchmark_vqe.py`
- **Description**: This script benchmarks the Variational Quantum Eigensolver (VQE) by measuring execution time and accuracy in estimating the ground state energy of a Hamiltonian. It visualizes the results for different numbers of qubits.

### Benchmarking Hybrid Quantum-Classical Algorithms
- **File**: `benchmark_hybrid.py`
- **Description**: This script benchmarks hybrid quantum-classical algorithms, specifically the Quantum Approximate Optimization Algorithm (QAOA). It measures execution time and solution quality for different numbers of layers in the QAOA circuit.

## Requirements

To run the benchmark scripts, ensure you have the following packages installed:

- Qiskit
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install qiskit numpy matplotlib
```

## Usage
To use any of the benchmark scripts, navigate to the benchmarks directory and execute the desired script using Python. For example, to run the Grover's algorithm benchmark:

```bash
1 python benchmark_grover.py
```

Each script will output the benchmark results to the console and may generate visualizations to help analyze the performance.

## Contributing
Contributions to the benchmark scripts are welcome! If you have suggestions for improvements or new benchmarks to implement, please open an issue or submit a pull request.

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
