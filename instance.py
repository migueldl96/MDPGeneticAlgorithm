import numpy as np


class MDPInstance:
    """
    Doc
    """

    def __init__(self, filename):
        # Read the instance data from file
        try:
            with open(filename, 'r') as f:
                lines = f.readline()
                # First line -> Complete set and subset (N and M)
                self._N, self._M = map(int, lines.strip().split(' '))

                # Init matrix distances
                self._distances = np.empty(shape=(self._N, self._N))

                # Rest of lines -> Distances between elements
                for line in f:
                    i, j, distance = map(float, line.strip().split(' '))
                    self._distances[int(i), int(j)] = distance
                    self._distances[int(j), int(i)] = distance

        except FileNotFoundError:
            print('File \'' + filename + '\' does not exist.')
            exit(-1)

    def get_n(self):
        return self._N

    def get_m(self):
        return self._M

    def get_distance(self, n1, n2):
        return self._distances[n1, n2]
