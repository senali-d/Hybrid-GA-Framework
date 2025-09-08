from src.problems.tsp import TravelingSalesmanProblem

TSP_NAME = "bayg29"
tsp_instance = TravelingSalesmanProblem(TSP_NAME)

def tsp_fitness(individual):
    return tsp_instance.fitness(individual),  # negative if minimizing
