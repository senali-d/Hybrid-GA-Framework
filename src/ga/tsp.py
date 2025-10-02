from src.problems.tsp import TravelingSalesmanProblem

# create instance
TSP_NAME = "bayg29"
tsp_instance = TravelingSalesmanProblem(TSP_NAME)


# fitness calculation
def tsp_fitness(individual):
    return (tsp_instance.fitness(individual),)  # negative if minimizing
