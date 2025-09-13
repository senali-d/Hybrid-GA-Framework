from src.problems.rosenbrock import RosenbrockProblem

rosenbrock = RosenbrockProblem()

def rosenbrock_fitness(individual):
    return rosenbrock.fitness(individual)
