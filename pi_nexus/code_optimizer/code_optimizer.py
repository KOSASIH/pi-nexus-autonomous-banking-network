import random

from deap import base, creator, tools


class CodeOptimizer:
    def __init__(self, code: str, fitness_function):
        self.code = code
        self.fitness_function = fitness_function
        self.population = [code] * 100

    def evolve_code(self, generations: int) -> str:
        for _ in range(generations):
            offspring = algorithms.varAnd(self.population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = [self.fitness_function(ind) for ind in offspring]
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            self.population = toolbox.select(offspring, k=len(self.population))
        return max(self.population, key=self.fitness_function)
