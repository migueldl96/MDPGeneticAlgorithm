import numpy as np

class MDPReplacement():
    """
    Doc - Abstract
    """
    def run_replacement(self, original_pop, new_pop):
        pass


class MDPWholeReplacement(MDPReplacement):
    """
    Replace the entire original population
    """
    def run_replacement(self, original_pop, new_pop):
        return new_pop


class MDPBestOfBoth(MDPReplacement):
    """
    Select the k best individuals from both populations
    """

    def run_replacement(self, original_pop, new_pop):

        # Concatenate populations
        whole_population = np.concatenate((original_pop, new_pop))

        # Replace
        pop_size = len(original_pop)
        new_population = np.sort(whole_population)[-pop_size:]

        return new_population


class MDPReplaceWorstOffspring(MDPReplacement):
    """
    Replace the worst offspring individual if the original best is
    better than the offspring best
    """
    def run_replacement(self, original_pop, new_pop):

        # Calculate best of both populations
        resulting_pop = new_pop
        best_original = max(original_pop)
        best_new = max(new_pop)

        # Is there a better individual in original population?
        if best_original.get_fitness() > best_new.get_fitness():
            index_worst_offspring = np.argmin(new_pop)
            resulting_pop[index_worst_offspring] = best_original

        return resulting_pop
