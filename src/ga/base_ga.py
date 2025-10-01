from deap import base, creator, tools, algorithms
import random
import array
import numpy as np

class BaseGA:
    def __init__(self, fitness_func, individual_size, chromosome_type="binary",
                 int_range=(0, 10), real_range=(0.0, 1.0),
                 population_size=100, ngen=100,
                 crossover_prob=0.8, mutation_prob=0.2,
                 hall_of_fame_size=1, maximize=True, seed=None):
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
        self.hall_of_fame_size = hall_of_fame_size
        self.maximize = maximize

        # DEAP setup
        weight = 1.0 if maximize else -1.0
        creator.create("FitnessType", base.Fitness, weights=(weight,))
        if chromosome_type in ["binary", "integer", "real"]:
            creator.create("Individual", list, fitness=creator.FitnessType)
        elif chromosome_type == "permutation":
            creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessType)
        else:
            raise ValueError("Unsupported chromosome type")

        self.toolbox = base.Toolbox()
        self._setup_encoding()
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individualCreator)

        self.toolbox.register("evaluate", self.fitness_func)
        self.toolbox.register("select", tools.selRoulette)

    def _setup_encoding(self):
        if self.chromosome_type == "binary":
            self.toolbox.register("attr_gene", random.randint, 0, 1)
            self.toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_gene, n=self.individual_size)
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/self.individual_size)

        elif self.chromosome_type == "integer":
            low, high = self.int_range
            self.toolbox.register("attr_gene", random.randint, low, high)
            self.toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_gene, n=self.individual_size)
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1/self.individual_size)

        elif self.chromosome_type == "permutation":
            self.toolbox.register("randomOrder", random.sample, range(self.individual_size), self.individual_size)
            self.toolbox.register("individualCreator", tools.initIterate, creator.Individual, self.toolbox.randomOrder)
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/self.individual_size)

        elif self.chromosome_type == "real":
            low, high = self.real_range
            self.toolbox.register("attr_gene", random.uniform, low, high)
            self.toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_gene, n=self.individual_size)
            self.toolbox.register("mate", tools.cxOnePoint)
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1/self.individual_size)

    def run(self):
        population = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(self.hall_of_fame_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        if self.maximize:
            stats.register("max", np.max)
            stats.register("avg", np.mean)
        else:
            stats.register("min", np.min)
            stats.register("avg", np.mean)

        population, logbook = algorithms.eaSimple(
            population, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
            ngen=self.ngen, stats=stats, halloffame=hof, verbose=True
        )
        return hof.items[0], hof.items[0].fitness.values, logbook
