from ga.base_ga import BaseGA
import seaborn as sns
import matplotlib.pyplot as plt
from config.setting import PROBLEMS, DEFAULT_GA_PARAMS

PROBLEM = "timetabling"  # options: "tsp", "knapsack", "nurses", "timetabling", "rosenbrock, "integer", "real"


def plot_fitness(max_fitness_values, mean_fitness_values, maximize=True):
    """
    Dynamically plot fitness evolution for GA runs.
    Automatically handles minimization or maximization.
    """
    sns.set_style("whitegrid")
    plt.figure()

    if maximize:
        main_label = "Max Fitness"
        avg_label = "Mean Fitness"
        main_color = "red"
        better_text = "higher"
    else:
        main_label = "Min Fitness"
        avg_label = "Mean Fitness"
        main_color = "blue"
        better_text = "lower"

    # Plot the curves
    plt.plot(max_fitness_values, color=main_color, label=main_label)
    plt.plot(mean_fitness_values, color="green", label=avg_label)

    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.title(
        f"{main_label} and {avg_label} over Generations\n(Better = {better_text})"
    )
    plt.legend()
    plt.show()


def main():
    cfg = PROBLEMS[PROBLEM]
    ga_params = DEFAULT_GA_PARAMS

    extra = cfg.get("extra_params", lambda: {})()

    ga = BaseGA(
        fitness_func=cfg["fitness_func"],
        individual_size=cfg["individual_size"],
        chromosome_type=cfg["chromosome_type"],
        population_size=ga_params["POPULATION_SIZE"],
        ngen=ga_params["MAX_GENERATIONS"],
        crossover_prob=ga_params["P_CROSSOVER"],
        mutation_prob=ga_params["P_MUTATION"],
        maximize=cfg["maximize"],
        **extra,
    )

    results = ga.run()

    if ga.maximize:
        fitness_curve = results["max_fitness_values"]
    else:
        fitness_curve = results["min_fitness_values"]

    mean_curve = results["mean_fitness_values"]

    # Plot
    plot_fitness(fitness_curve, mean_curve, maximize=ga.maximize)


if __name__ == "__main__":
    main()
