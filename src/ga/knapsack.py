from src.problems.knapsack import Knapsack01Problem

# create instance
knapsack = Knapsack01Problem()

def knapsack_fitness(individual):
    # individual is a list of 0/1
    return knapsack.fitness(individual),   # tuple for DEAP
