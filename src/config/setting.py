from src.ga.tsp import tsp_fitness, tsp_instance
from src.ga.knapsack import knapsack_fitness, knapsack

DEFAULT_GA_PARAMS = {
    # "POPULATION_SIZE": 200,
    # "MAX_GENERATIONS": 300,
    "POPULATION_SIZE": 50,
    "MAX_GENERATIONS": 50,

    "HALL_OF_FAME_SIZE": 1,
    "P_CROSSOVER": 0.9,
    "P_MUTATION": 0.1,
}

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
        "plot_func": knapsack.plotData,
        "stats": ("max", "avg"),
    },
}
