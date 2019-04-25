from evaluator import MDPEvaluator
import numpy as np


class MDPSolution:
    def __init__(self, instance):
        # Solution attributes
        self._instance = instance
        self._N = instance.get_n()
        self._M = instance.get_m()

        # Solution information
        self._fitness = 0
        self._fitness_assigned = False
        self._solution_assigned = False

        # Solution integer representation
        self._solution = np.empty(shape=(self._M,), dtype=int)

    def calculate_fitness(self):
        self._fitness = MDPEvaluator.evaluate(self, self._instance)
        self._fitness_assigned = True

    def set_element(self, position, element):
        self._solution[position] = element
        self._fitness_assigned = False

    def get_element(self, position):
        return self._solution[position]

    def get_fitness(self):
        return self._fitness

    def set_solution(self, solution):
        assert len(solution) == self._M
        assert len(np.unique(solution)) == self._M
        self._solution = solution
        self.calculate_fitness()
        self._solution_assigned = True

    def get_solution(self):
        return self._solution
