from geneticAlgorithm import MDPGeneticAlgorithm
from instance import MDPInstance
from selector import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Parameters
    generations = 200

    # Instance
    instance = MDPInstance('instances/GKD-c_1_n500_m50.txt')

    # Algorithm
    alg = MDPGeneticAlgorithm(instance, generations=generations, elitism=True,
                              selector=MDPTournamentSelection(size=5, p=0.2),
                              mutation_prob=0.1)
    moment_results, generation_results = alg.run()

    # Plot the results
    plt.plot(range(generations), [moment_results[i].get_fitness() for i in range(generations)], 'b-')
    plt.plot(range(generations), [generation_results[i].get_fitness() for i in range(generations)], 'r.')

    plt.show()



