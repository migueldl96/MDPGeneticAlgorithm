import numpy as np


class MDPSelector:
    """
    Doc - Abstract class
    """
    def run_selection(self, population):
        pass


class MDPRouletteSelection(MDPSelector):
    """
    Doc
    """
    def run_selection(self, population):
        fitness_array = [population[i].get_fitness() for i in range(0, len(population))]
        max_value = sum(fitness_array)
        current_value = 0
        limit = np.random.uniform(0, max_value)
        for individual in population:
            current_value = current_value + individual.get_fitness()
            if current_value > limit:
                return individual


class MDPTournamentSelection(MDPSelector):
    """
    Doc
    """
    def __init__(self, size=3, p=0.3):
        self._size = size
        self._p = p

    def run_selection(self, population):
        # Sort population to calculate probability of been chosen
        sorted_population = np.sort(population)[::-1]
        probabilities = np.array([self._p*(1.0-self._p)**i for i in range(len(population))])
        probabilities /= probabilities.sum()     # Normalize for avoid non-1 sum

        # Choice some of them
        candidates = np.random.choice(sorted_population, self._size, p=probabilities)

        # Return winner
        return max(candidates)
