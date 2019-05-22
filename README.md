# MDPGeneticAlgorithm
Python implemention of genetic algorithm for solving Maximum Diversity Problem. 

## Introduction
This repo contains a Genetic Algorithm (GA) approach for solving the Maximum Diversity Problem. This problem consists of determining a subset *M* of a given cardinality from a set *N* of elements, in such away that the sum of the pair-wise distances between the elements of *M* is the maximum possible.

This problem is considered **NP-hard**, meaning that it does not exist an algorithm for getting the exact solution in a polinomial time. These kind of problems can be faced with soft computing techniques, which return a acceptable solution in a affordable time.

## Solution representation
Each individual (solution) in the population is represented by an integer 1D array of size *M*, where each element represent a chosen element form the *N* set.

## Behavior
Once a initial population of *P* individuals is initiated, a generation of the implemented GA works as follows:

    1 - Init new population ramdomly
    2 - Selection: 2*P individuals of the populations are selected to create offspring. Two selection mechanism are implemented: Roullete and Tournament.
    3 - Cross-over: Each couple generate a new individual crossing its elements (genes). The commoms gens of the parents are transmited entirely to the new individual, non-commons genes are transmited over 50% probability for each parent.
    4 - Mutation: Each gene of the new individual is randomly change for another value with a low probabily.
    5 - Replacement: New and old population are combined in some way. Three replacement operators are build: WholeReplacement, BestOfBoth and ReplaceWorstOffspring

## How to run
*runGeneticAlgorithm.py* file is prepared to run the GeneticAlgorithm. The parameters (number os generations, population size, mutation prob...) can be modified. This script will save the results of the execution in *results* directory.

## Notes
*instances* directory contains some instance examples.

