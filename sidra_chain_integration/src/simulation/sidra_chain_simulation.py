# sidra_chain_simulation.py
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SidraChainSimulation:
    def __init__(self):
        # Initialize simulation parameters
        self.mass = 10.0
        self.spring_constant = 100.0
        self.damping_coefficient = 5.0
        self.time_step = 0.01
        self.simulation_time = 10.0

    def equations_of_motion(self, state, t):
        # Define the equations of motion for the system
        x, v = state
        a = (-self.spring_constant * x - self.damping_coefficient * v) / self.mass
        return [v, a]

    def simulate_system(self, initial_conditions):
        # Simulate the system using scipy's odeint
        t = np.arange(0, self.simulation_time, self.time_step)
        state0 = initial_conditions
        solution = odeint(self.equations_of_motion, state0, t)
        return t, solution

    def plot_results(self, t, solution):
        # Plot the simulation results
        plt.plot(t, solution[:, 0], label='Position (x)')
        plt.plot(t, solution[:, 1], label='Velocity (v)')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.title('Sidra Chain Simulation Results')
        plt.legend()
        plt.show()

    def run_simulation(self, initial_conditions):
        # Run the simulation and plot the results
        t, solution = self.simulate_system(initial_conditions)
        self.plot_results(t, solution)

if __name__ == "__main__":
    simulation = SidraChainSimulation()
    initial_conditions = [1.0, 0.0]  # Initial position and velocity
    simulation.run_simulation(initial_conditions)
