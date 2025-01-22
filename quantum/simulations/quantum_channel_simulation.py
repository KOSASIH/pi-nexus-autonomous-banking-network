# quantum_channel_simulation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import plot_bloch_multivector, plot_state_qsphere

def apply_depolarizing_channel(state, p):
    """
    Apply a depolarizing channel to a quantum state.

    Parameters:
    - state: The input quantum state (as a Statevector).
    - p: The depolarizing probability (0 <= p <= 1).

    Returns:
    - DensityMatrix: The resulting state after applying the depolarizing channel.
    """
    # Calculate the density matrix of the input state
    density_matrix = DensityMatrix(state)

    # Depolarizing channel operation
    identity = np.eye(2)  # Identity operator
    depolarizing_matrix = (1 - p) * density_matrix + (p / 3) * (density_matrix.trace() * identity)

    return depolarizing_matrix

def apply_amplitude_damping_channel(state):
    """
    Apply an amplitude damping channel to a quantum state.

    Parameters:
    - state: The input quantum state (as a Statevector).

    Returns:
    - DensityMatrix: The resulting state after applying the amplitude damping channel.
    """
    # Calculate the density matrix of the input state
    density_matrix = DensityMatrix(state)

    # Amplitude damping channel operation
    damping_matrix = np.array([[1, 0], [0, np.sqrt(1)]])
    damping_matrix_dagger = np.array([[1, 0], [0, 0]])

    # Apply the amplitude damping channel
    new_density_matrix = damping_matrix @ density_matrix @ damping_matrix_dagger

    return DensityMatrix(new_density_matrix)

def simulate_channel(state, channel_type, p=None):
    """
    Simulate the effect of a quantum channel on a given quantum state.

    Parameters:
    - state: The input quantum state (as a Statevector).
    - channel_type: The type of channel ('depolarizing' or 'amplitude_damping').
    - p: The depolarizing probability (only needed for depolarizing channel).

    Returns:
    - DensityMatrix: The resulting state after applying the specified channel.
    """
    if channel_type == 'depolarizing':
        if p is None:
            raise ValueError("Depolarizing probability p must be provided.")
        return apply_depolarizing_channel(state, p)
    elif channel_type == 'amplitude_damping':
        return apply_amplitude_damping_channel(state)
    else:
        raise ValueError("Invalid channel type. Choose 'depolarizing' or 'amplitude_damping'.")

def visualize_state(state, title="Quantum State"):
    """
    Visualize the quantum state on the Bloch sphere and Q-sphere.

    Parameters:
    - state: The quantum state (as a DensityMatrix).
    - title: Title for the visualization.
    """
    # Plot Bloch vector representation
    plot_bloch_multivector(state)
    plt.title(f"{title} - Bloch Sphere")
    plt.show()

    # Plot Q-sphere representation
    plot_state_qsphere(state)
    plt.title(f"{title} - Q-Sphere")
    plt.show()

if __name__ == "__main__":
    # Example usage
    # Create an initial quantum state (|0⟩ + |1⟩) / sqrt(2)
    initial_state = Statevector.from_dict({'00': 0.7071, '01': 0.7071})

    # Simulate depolarizing channel
    p = 0.1  # Depolarizing probability
    depolarized_state = simulate_channel(initial_state, 'depolarizing', p)
    print("Depolarized State:")
    print(dep polarized_state)

    # Visualize the depolarized state
    visualize_state(depolarized_state, title="Depolarized State")

    # Simulate amplitude damping channel
    damped_state = simulate_channel(initial_state, 'amplitude_damping')
    print("Damped State:")
    print(damped_state)

    # Visualize the damped state
    visualize_state(damped_state, title="Damped State")
