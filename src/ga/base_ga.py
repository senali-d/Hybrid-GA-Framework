from deap import base, creator, tools, algorithms
import random
import array
import numpy as np


class BaseGA:
    def __init__(
        self,
        fitness_func,
        individual_size,
        chromosome_type="binary",
        int_range=(0, 10),
        real_range=(0.0, 1.0),
        population_size=100,
        ngen=100,
        crossover_prob=0.8,
        mutation_prob=0.2,
        maximize=True,
        seed=None,
    ):
        """
        Generic Genetic Algorithm using DEAP.
        Supports binary, integer, permutation, and real encodings.
        """
        if seed is not None:
            random.seed(seed)

        self.fitness_func = fitness_func
        self.individual_size = individual_size
        self.chromosome_type = chromosome_type
        self.int_range = int_range
        self.real_range = real_range
        self.population_size = population_size
        self.ngen = ngen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.maximize = maximize

        # DEAP setup
        weight = 1.0 if maximize else -1.0
        creator.create("FitnessType", base.Fitness, weights=(weight,))
        if chromosome_type in ["binary", "integer", "real"]:
            creator.create("Individual", list, fitness=creator.FitnessType)
        elif chromosome_type == "permutation":
            creator.create(
                "Individual", array.array, typecode="i", fitness=creator.FitnessType
            )
        else:
            raise ValueError("Unsupported chromosome type")

        self.toolbox = base.Toolbox()
        self._setup_encoding()
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individualCreator
        )

        self.toolbox.register("evaluate", self.fitness_func)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _setup_encoding(self):
        if self.chromosome_type == "binary":
            self.toolbox.register("attr_gene", random.randint, 0, 1)
            self.toolbox.register(
                "individualCreator",
                tools.initRepeat,
                creator.Individual,
                self.toolbox.attr_gene,
                n=self.individual_size,
            )
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)

        elif self.chromosome_type == "integer":
            low, high = self.int_range
            self.toolbox.register("attr_gene", random.randint, low, high)
            self.toolbox.register(
                "individualCreator",
                tools.initRepeat,
                creator.Individual,
                self.toolbox.attr_gene,
                n=self.individual_size,
            )
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)

        elif self.chromosome_type == "permutation":
            self.toolbox.register(
                "randomOrder",
                random.sample,
                range(self.individual_size),
                self.individual_size,
            )
            self.toolbox.register(
                "individualCreator",
                tools.initIterate,
                creator.Individual,
                self.toolbox.randomOrder,
            )
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)

        elif self.chromosome_type == "real":
            low, high = self.real_range
            self.toolbox.register("attr_gene", random.uniform, low, high)
            self.toolbox.register(
                "individualCreator",
                tools.initRepeat,
                creator.Individual,
                self.toolbox.attr_gene,
                n=self.individual_size,
            )
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)

    def run(self):
        # --- Create initial population ---
        population = self.toolbox.population(n=self.population_size)
        generation_counter = 0

        # --- Evaluate initial population ---
        fitness_values = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitness_values):
            ind.fitness.values = fit

        # --- Decide objective direction dynamically ---
        if self.maximize:
            best_func = np.max
            compare_func = max
            best_label = "Max"
        else:
            best_func = np.min
            compare_func = min
            best_label = "Min"

        best_fitness_values = []
        mean_fitness_values = []

        # --- Evolutionary loop ---
        while generation_counter < self.ngen:
            generation_counter += 1

            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitness_values = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitness_values):
                ind.fitness.values = fit

            # Replace old population
            population[:] = offspring

            # --- Gather statistics ---
            fitness_values = [ind.fitness.values[0] for ind in population]
            best_fitness = best_func(fitness_values)
            mean_fitness = np.mean(fitness_values)

            best_fitness_values.append(best_fitness)
            mean_fitness_values.append(mean_fitness)

            print(
                f"- Generation {generation_counter}: {best_label} Fitness = {best_fitness}, Avg Fitness = {mean_fitness}")

            # --- Find and print best individual dynamically ---
            best_index = fitness_values.index(compare_func(fitness_values))
            print("Best Individual = ", *population[best_index], "\n")

        # --- Return dynamically labeled result ---
        if self.maximize:
            return {"max_fitness_values": best_fitness_values, "mean_fitness_values": mean_fitness_values}
        else:
            return {"min_fitness_values": best_fitness_values, "mean_fitness_values": mean_fitness_values}
