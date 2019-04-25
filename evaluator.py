from itertools import product


class MDPEvaluator:
    """
    Doc
    """
    @staticmethod
    def evaluate(solution, instance):
        fitness = 0

        # Calculate total distance
        for i in range (0, instance.get_m()):
            for j in range(i+1, instance.get_m()):
                fitness = fitness + instance.get_distance(solution.get_element(i),
                                                          solution.get_element(j))

        return fitness
