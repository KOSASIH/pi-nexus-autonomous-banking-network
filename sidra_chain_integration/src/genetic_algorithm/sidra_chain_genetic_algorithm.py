# sidra_chain_genetic_algorithm.py
import random
import numpy as np

class SidraChainGeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        return population

    def create_individual(self):
        # Create a random individual with 10 genes
        individual = [random.randint(0, 1) for _ in range(10)]
        return individual

    def fitness_function(self, individual):
        # Calculate the fitness of an individual
        fitness = sum(individual)
        return fitness

    def selection(self):
        # Select the fittest individuals for reproduction
        fitnesses = [self.fitness_function(individual) for individual in self.population]
        sorted_indices = np.argsort(fitnesses)[::-1]
        selected_individuals = [self.population[i] for i in sorted_indices[:int(self.population_size/2)]]
        return selected_individuals

    def crossover(self, parent1, parent2):
        # Perform crossover (recombination) of two parents
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, individual):
        # Perform mutation on an individual
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def evolve(self):
        for _ in range(self.generations):
            selected_individuals = self.selection()
            offspring = []
            while len(offspring) < self.population_size:
                parent1, parent2 = random.sample(selected_individuals, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.extend([child1, child2])
            self.population = offspring

    def get_fittest_individual(self):
        fitnesses = [self.fitness_function(individual) for individual in self.population]
        fittest_index = np.argmax(fitnesses)
        return self.population[fittest_index]

# Example usage
ga = SidraChainGeneticAlgorithm(population_size=100, generations=50, mutation_rate=0.01)
ga.evolve()
fittest_individual = ga.get_fittest_individual()
print("Fittest individual:", fittest_individual)
