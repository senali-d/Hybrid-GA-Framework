from src.problems.nurses import NurseSchedulingProblem

HARD_CONSTRAINT_PENALTY = 10

# create instance
nsp = NurseSchedulingProblem(HARD_CONSTRAINT_PENALTY)


# fitness calculation
def nurses_fitness(individual):
    return (nsp.getCost(individual),)
