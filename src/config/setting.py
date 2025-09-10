from src.ga.tsp import tsp_fitness, tsp_instance
from src.ga.knapsack import knapsack_fitness, knapsack
from src.ga.nurses import nurses_fitness, nsp

DEFAULT_GA_PARAMS = {
    # tsp
    # "POPULATION_SIZE": 200,
    # "MAX_GENERATIONS": 300,

    # knapsak
    # "POPULATION_SIZE": 50,
    # "MAX_GENERATIONS": 50,

    # nursing
    "POPULATION_SIZE": 300,
    "MAX_GENERATIONS": 200,
    "HALL_OF_FAME_SIZE": 30,

    # "HALL_OF_FAME_SIZE": 1,
    "P_CROSSOVER": 0.9,
    "P_MUTATION": 0.1,
    "SEED": 42,
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
    "nurses": {
        "fitness_func": nurses_fitness,
        "individual_size": len(nsp),
        "chromosome_type": "binary",
        "maximize": False,
        "plot_func": nsp.plotData,
        "stats": ("min", "avg"),
    },
}
