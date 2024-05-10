# Import necessary libraries
import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from qiskit.optimization import QuadraticProgram

# Define a function to optimize a portfolio using quantum computing
def optimize_portfolio(stocks, weights, risk_tolerance):
    # Define the quadratic program
    qp = QuadraticProgram()
    qp.binary_var_list(stocks, name='x')
    qp.minimize(linear={stocks[i]: weights[i] for i in range(len(stocks))})
    qp.subject_to(LinearConstraint(sense='LE', rhs=risk_tolerance, coeffs={stocks[i]: weights[i] for i in range(len(stocks))}))
    
    # Convert the quadratic program to a QUBO problem
    qubo = qp.to_qubo()
    
    # Create a quantum circuit to solve the QUBO problem
    qc = QuantumCircuit(len(stocks))
    qc.h(range(len(stocks)))
    qc.barrier()
    qc.x(range(len(stocks)))
    qc.barrier()
    qc.measure_all()
    
    # Execute the circuit on a simulator
    simulator = AerSimulator()
    job = execute(qc, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    # Extract the optimal solution from the counts
    optimal_solution = max(counts, key=counts.get)
    optimal_portfolio = [stocks[i] for i, x in enumerate(optimal_solution) if x == '1']
    
    return optimal_portfolio

# Example usage
stocks = ['AAPL', 'GOOG', 'MSFT']
weights = [0.3, 0.4, 0.3]
risk_tolerance = 0.5
optimal_portfolio = optimize_portfolio(stocks, weights, risk_tolerance)
print("Optimal portfolio:", optimal_portfolio)
