from src.ga.tsp import tsp_fitness, tsp_instance
from src.ga.knapsack import knapsack_fitness, knapsack
from src.ga.nurses import nurses_fitness, nsp
from src.ga.timetabling import timetable_fitness, timetable_instance
from src.ga.rosenbrock import rosenbrock_fitness, rosenbrock
from src.problems.timetabling import TimetablingProblem

DEFAULT_GA_PARAMS = {
    # tsp
    # "POPULATION_SIZE": 300,
    # "MAX_GENERATIONS": 200,
    # rosenbrock
    "POPULATION_SIZE": 50,
    "MAX_GENERATIONS": 100,
    # knapsack
    # "POPULATION_SIZE": 50,
    # "MAX_GENERATIONS": 50,
    # nursing
    # "POPULATION_SIZE": 300,
    # "MAX_GENERATIONS": 200,
    # timetabling
    # "POPULATION_SIZE": 150,
    # "MAX_GENERATIONS": 600,
    "P_CROSSOVER": 0.6,
    "P_MUTATION": 0.1,
    "SEED": 42,
}

HARD_CONSTRAINT_PENALTY = 10

PROBLEMS = {
    "tsp": {
        "fitness_func": tsp_fitness,
        "individual_size": len(tsp_instance),
        "chromosome_type": "permutation",
        "maximize": False,
        "plot_func": tsp_instance.plotData,
        "stats": ("min", "avg"),
    },
    "knapsack": {
        "fitness_func": knapsack_fitness,
        "individual_size": len(knapsack),
        "chromosome_type": "binary",
        "maximize": True,
        "plot_func": knapsack.printItems,
        "stats": ("max", "avg"),
    },
    "nurses": {
        "fitness_func": nurses_fitness,
        "individual_size": len(nsp),
        "chromosome_type": "binary",
        "maximize": False,
        "plot_func": nsp.printScheduleInfo,
        "stats": ("min", "avg"),
    },
    "timetabling": {
        "fitness_func": timetable_fitness,
        "individual_size": len(timetable_instance),
        "chromosome_type": "integer",
        "maximize": False,
        "plot_func": timetable_instance.printSchedule,
        "stats": ("min", "avg"),
        "extra_params": lambda: {
            "int_range": (
                0,
                TimetablingProblem(HARD_CONSTRAINT_PENALTY).numRooms
                * TimetablingProblem(HARD_CONSTRAINT_PENALTY).numTimeslots
                - 1,
            )
        },
    },
    "rosenbrock": {
        "fitness_func": rosenbrock_fitness,
        "individual_size": len(rosenbrock),
        "chromosome_type": "real",
        "maximize": False,
        "plot_func": rosenbrock.printSolution,
        "stats": ("min", "avg"),
        "real_range": (-5, 5),
    },
}
