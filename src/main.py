from ga.base_ga import BaseGA
import seaborn as sns
import matplotlib.pyplot as plt
from config.setting import PROBLEMS, DEFAULT_GA_PARAMS

PROBLEM = "knapsack"  # options: "tsp", "knapsack", "integer", "real"

def main():
    cfg = PROBLEMS[PROBLEM]
    ga_params = DEFAULT_GA_PARAMS

    ga = BaseGA(
        fitness_func=cfg["fitness_func"],
        individual_size=cfg["individual_size"],
        chromosome_type=cfg["chromosome_type"],
        population_size=ga_params["POPULATION_SIZE"],
        ngen=ga_params['MAX_GENERATIONS'],
        crossover_prob=ga_params['P_CROSSOVER'],
        mutation_prob=ga_params["P_MUTATION"],
        maximize=cfg["maximize"],
        hall_of_fame_size=ga_params["HALL_OF_FAME_SIZE"],
        seed=42,
    )

    best, fitness, logbook = ga.run()

    print("Best solution:", best)
    print("Fitness:", fitness[0])

    # Plot the solution if supported
    if cfg["plot_func"] is not None:
        cfg["plot_func"](best)

    # Extract statistics
    stat1, stat2 = cfg["stats"]
    values1, values2 = logbook.select(stat1, stat2)

    # Plot statistics
    plt.figure()
    sns.set_style("whitegrid")
    plt.plot(values1, color='red')
    plt.plot(values2, color='green')
    plt.xlabel('Generation')
    plt.ylabel(f"{stat1.capitalize()} / {stat2.capitalize()} Fitness")
    plt.title(f"{stat1.capitalize()} and {stat2.capitalize()} Fitness over Generations")
    plt.show()

if __name__ == "__main__":
    main()
