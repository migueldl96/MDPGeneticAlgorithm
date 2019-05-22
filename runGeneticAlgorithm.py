from classes.geneticAlgorithm import MDPGeneticAlgorithm
from classes.instance import MDPInstance
from classes.selector import *
from classes.replacement import *
import matplotlib.pyplot as plt
import os


def save_results(sel_name, rep_name, n_gen, mut_p, ins_name, best, generation, mean):

    # Get float results
    best_float = [best[i].get_fitness() for i in range(len(best))]
    generation_float = [generation[i].get_fitness() for i in range(len(generation))]

    # Merge results into a single array
    results = np.array([best_float, generation_float, mean])

    # Experiment name
    exp_name = ins_name + '_gen' + str(n_gen) + '_mut' + str(mut_p) + '_sel' + sel_name + '_rep' + rep_name

    # Path
    result_path = 'results/' + exp_name + '/'
    os.mkdir(path=result_path)

    np.savetxt(result_path + 'results.csv', results.transpose(), delimiter=',', fmt='%f', header="Best,Current,Mean")

    # Save plot
    # Lines style
    best_line = plt.plot(range(n_gen), [best[i].get_fitness() for i in range(n_gen)], label='Best')
    generation_line = plt.plot(range(n_gen), [generation[i].get_fitness() for i in range(n_gen)], label='Current')
    mean_line = plt.plot(range(n_gen), [mean[i] for i in range(n_gen)], label='Mean')

    plt.setp(best_line, linewidth=1, linestyle='-', color='b')
    plt.setp(generation_line, linewidth=1, linestyle=':', color='r')
    plt.setp(mean_line, linewidth=1, linestyle='--', color='g')

    plt.legend(loc='upper left')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    plt.savefig(result_path + exp_name + '-plot.eps', format='eps')

    plt.clf()


if __name__ == '__main__':
    # Parameters
    n_generations = 500
    pop_size = 50
    mutation_p = 0.05
    selector = MDPRouletteSelection()
    replacement = MDPReplaceWorstOffspring()
    instance_name = 'GKD-c_4_n500_m50'

    # Instance
    instance = MDPInstance('instances/' + instance_name + '.txt')

    # Algorithm
    alg = MDPGeneticAlgorithm(instance,
                              population_size=pop_size,
                              n_generations=n_generations,
                              selector=selector,
                              replacement=replacement,
                              mutation_prob=mutation_p)

    current_results, generation_results, mean_results = alg.run()

    # Save results
    save_results(selector.__class__.__name__,
                 replacement.__class__.__name__,
                 n_generations, mutation_p, instance_name,
                 current_results, generation_results, mean_results)





