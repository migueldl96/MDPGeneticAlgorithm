from solGenerator import MDPSolGenerator
from solution import MDPSolution
from selector import *
from replacement import *
from copy import deepcopy
import numpy as np


class MDPGeneticAlgorithm:
    """
    Doc
    """
    def __init__(self, instance, selector=MDPRouletteSelection(), population_size=50,
                 generations=50, mutation_prob=0.1, replacement=MDPWholeReplacement()):

        # Algorithm attributes
        self._instance = instance
        self._population_size = population_size
        self._generations = generations
        self._mutation_prob = mutation_prob
        self._selector = selector
        self._replacement = replacement

        # Algorithm data
        self._population = None
        self._best_solution = MDPSolution(instance)

        # Results structures
        self._best_solutions_until_now = np.ndarray(shape=(self._generations,), dtype=np.object)
        self._best_solutions_generation = np.ndarray(shape=(self._generations,), dtype=np.object)
        self._mean_solutions_generation = np.ndarray(shape=(self._generations,), dtype=float)

    def init_population(self):
        self._population = np.ndarray(shape=(self._population_size,), dtype=np.object)

        # Generate random valid solutions
        for i in range(0, self._population_size):
            self._population[i] = MDPSolGenerator.random_generator(instance=self._instance)

        self.save_best_solution(0)

    def save_best_solution(self, generation):

        # Save best solution
        best_population_solution = max(self._population)

        if generation == 0:
            self._best_solutions_until_now[generation] = best_population_solution
        else:
            if best_population_solution.get_fitness() > self._best_solutions_until_now[generation-1].get_fitness():
                self._best_solutions_until_now[generation] = deepcopy(best_population_solution)
            else:
                self._best_solutions_until_now[generation] = self._best_solutions_until_now[generation-1]

        self._best_solutions_generation[generation] = best_population_solution

        # Mean generation solution
        self._mean_solutions_generation[generation] = np.mean(self._population)

    def roulette_selector(self):
        fitness_array = [self._population[i].get_fitness() for i in range(0, self._population_size)]
        max_value = sum(fitness_array)
        current_value = 0
        limit = np.random.uniform(0, max_value)
        for individual in self._population:
            current_value = current_value + individual.get_fitness()
            if current_value > limit:
                return individual

    def selection(self):
        candidates_size = 2*self._population_size
        canditates = np.ndarray(shape=(candidates_size,), dtype=np.object)

        # Roulette selection
        for i in range(0, candidates_size):
            canditates[i] = self._selector.run_selection(self._population)

        return canditates

    def cross_over(self, sol1, sol2):
        new_individual = MDPSolution(self._instance)

        # Cross non-comon elements to generate always valid solutions
        common_gens = np.intersect1d(sol1.get_solution(), sol2.get_solution())
        candidates_gens_sol1 = np.setdiff1d(sol1.get_solution(), sol2.get_solution())
        candidates_gens_sol2 = np.setdiff1d(sol2.get_solution(), sol1.get_solution())

        # Combine candidates with probability 0.5
        prob = 0.5
        new_solution = np.ndarray(shape=(len(candidates_gens_sol1),), dtype=int)
        for i in range(len(candidates_gens_sol1)):
            rand = np.random.uniform()
            if rand < 0.5:
                new_solution[i] = candidates_gens_sol1[i]
            else:
                new_solution[i] = candidates_gens_sol2[i]

        # Add common part
        new_solution = np.concatenate((new_solution, common_gens))

        # Construct new individual
        new_individual.set_solution(new_solution)

        return new_individual

    def mutate(self, sol):
        mutated_individual = deepcopy(sol)

        # Function for random mutation over possible values
        def random_mutation(sol):
            complete_set = np.array(range(self._instance.get_n()))
            candidates = np.setdiff1d(complete_set, sol.get_solution())

            return np.random.choice(candidates, 1)

        # Mutates gens
        for i in range(self._instance.get_m()):
            if np.random.uniform() < self._mutation_prob:
                mutated_individual.set_element(i, random_mutation(mutated_individual))

        return mutated_individual

    def run(self):
        # 1 - Init population
        if self._population is None:
            self.init_population()

        # Generations
        for i in range(1, self._generations):
            # Init new population
            new_population = np.ndarray(shape=(self._population_size,), dtype=np.object)

            # 2 - Selection
            candidates = self.selection()

            # 3 - Crossover
            for j in range(0, len(new_population)):
                new_population[j] = self.cross_over(candidates[j], candidates[(j*2)+1])

            # 4 - Mutation
            for j in range(0, len(new_population)):
                new_population[j] = self.mutate(new_population[j])

            # 5 - Population replacement
            self._population = self._replacement.run_replacement(self._population, new_population)

            # Save best results
            self.save_best_solution(i)

            print(i)

        # Return fitness evolution
        return self._best_solutions_until_now, self._best_solutions_generation, self._mean_solutions_generation
