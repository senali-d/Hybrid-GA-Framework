from src.problems.rosenbrock import RosenbrockProblem

# create instance
rosenbrock = RosenbrockProblem()


# fitness calculation
def rosenbrock_fitness(individual):
    return rosenbrock.fitness(individual)
