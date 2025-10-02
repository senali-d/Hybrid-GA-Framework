from src.problems.timetabling import TimetablingProblem

HARD_CONSTRAINT_PENALTY = 10

# create instance
timetable_instance = TimetablingProblem(HARD_CONSTRAINT_PENALTY)


# fitness calculation
def timetable_fitness(individual):
    return (timetable_instance.getCost(individual),)
