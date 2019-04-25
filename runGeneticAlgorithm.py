from geneticAlgorithm import MDPGeneticAlgorithm
from instance import MDPInstance


if __name__ == '__main__':
    instance = MDPInstance('instances/MDG-a_30_n2000_m200.txt')
    alg = MDPGeneticAlgorithm(instance, generations=100, elitism=True)
    alg.run()
    pass
