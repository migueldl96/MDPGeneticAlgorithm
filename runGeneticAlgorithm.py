from geneticAlgorithm import MDPGeneticAlgorithm
from instance import MDPInstance
from selector import *
from replacement import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Parameters
    experiment_name = ''
    generations = 350
    mutation_p = 0.05

    # Instance
    instance = MDPInstance('instances/GKD-c_9_n500_m50.txt')

    # Algorithm
    alg = MDPGeneticAlgorithm(instance, generations=generations,
                              selector=MDPRouletteSelection(),
                              replacement=MDPBestOfBoth(),
                              mutation_prob=mutation_p)

    moment_results, generation_results, mean_results = alg.run()

    # Plot the results
    plt.plot(range(generations), [moment_results[i].get_fitness() for i in range(generations)], 'b-')
    plt.plot(range(generations), [generation_results[i].get_fitness() for i in range(generations)], 'r.')
    plt.plot(range(generations), [mean_results[i] for i in range(generations)], 'g+')

    print(generation_results[-1].get_fitness())

    plt.show()



