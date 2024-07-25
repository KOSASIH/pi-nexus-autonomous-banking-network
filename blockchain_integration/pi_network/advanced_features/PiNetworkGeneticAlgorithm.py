# Importing necessary libraries
import random
import numpy as np

# Class for genetic algorithm
class PiNetworkGeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()

    # Function to initialize the population
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(-1, 1) for _ in range(10)]  # 10-dimensional vector
            population.append(individual)
        return population

    # Function to evaluate the fitness of an individual
    def evaluate_fitness(self, individual):
        # calculate the fitness of the individual
        fitness = np.sum(individual**2)
        return fitness

    # Function to select parents
    def select_parents(self):
        parents = []
        for _ in range(2):
            parent = random.choice(self.population)
            parents.append(parent)
        return parents

    # Function to crossover
    def crossover(self, parents):
        child = []
        for i in range(len(parents[0])):
            if random.random() < self.crossover_rate:
                child.append((parents[0][i] + parents[1][i]) / 2)
            else:
                child.append(random.choice([parents[0][i], parents[1][i]]))
        return child

    # Function to mutate
    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += random.uniform(-0.1, 0.1)
        return individual

    # Function to evolve the population
    def evolve(self):
        new_population = []
        while len(new_population) < self.population_size:
            parents = self.select_parents()
            child = self.crossover(parents)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    # Function to get the fittest individual
    def get_fittest_individual(self):
        fittest_individual = max(self.population, key=self.evaluate_fitness)
        return fittest_individual

# Example usage
ga = PiNetworkGeneticAlgorithm(population_size=100, mutation_rate=0.01, crossover_rate=0.5)
for _ in range(100):
    ga.evolve()
fittest_individual = ga.get_fittest_individual()
print(fittest_individual)
