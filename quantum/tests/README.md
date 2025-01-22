# Unit Tests for Quantum Components

This directory contains unit tests for various quantum components implemented in the project. The tests ensure the correctness and robustness of the quantum algorithms and functionalities.

## Test Files

- **test_grover.py**: Tests for Grover's algorithm, including circuit creation and execution.
- **test_shor.py**: Tests for Shor's algorithm, verifying the factorization of numbers.
- **test_qkd.py**: Tests for Quantum Key Distribution (QKD), checking the generated key length.
- **test_variational.py**: Tests for variational algorithms, ensuring the execution of VQE.
- **test_noise_simulation.py**: Tests for noise simulation, validating the creation and application of noise models.
- **test_hybrid_quantum.py**: Tests for hybrid quantum-classical algorithms, checking the success of the algorithm.

## Running Tests

To run the tests, navigate to the `tests` directory and execute the following command:

```bash
python -m unittest discover
```

This command will discover and run all the test cases in the directory.

## Requirements
Ensure that you have the necessary dependencies installed, including Qiskit and any other required libraries.

## Contributing
Contributions to the test suite are welcome! If you have suggestions for additional tests or improvements, please open an issue or submit a pull request.
