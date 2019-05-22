from classes.solution import MDPSolution
import numpy as np

class MDPSolGenerator:

    @staticmethod
    def random_generator(instance):
        random_solution = MDPSolution(instance)
        N = instance.get_n()
        M = instance.get_m()
        solution = np.random.permutation(N)[0:M]

        random_solution.set_solution(solution)
        return random_solution
