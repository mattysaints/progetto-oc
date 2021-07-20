import numpy as np


class TSP:
    def __init__(self, cost_mat: np.array):
        """
        Class that represents an instance of the symmetric TSP

        :param cost_mat: upper triangular matrix of costs
        """
        if cost_mat.shape[0] != cost_mat.shape[1]:
            raise ValueError('Cost matrix must be an upper triangular matrix')

        if not np.alltrue(np.tril(cost_mat) == 0):
            raise ValueError('Cost matrix must be an upper triangular matrix')

        self.cost_mat = cost_mat
        self.num_cities = cost_mat.shape[0]

    def cost(self, i, j):
        """
        Returns the cost of an edge (i,j)

        :return: cost of the edge (i,j)
        """
        if i >= j:
            i, j = j, i
        return self.cost_mat[i, j]

    def tour_cost(self, tour: np.array):
        """
        Returns the cost of a tour

        :param tour: numpy.array representing a permutation of the cities, which are identified by integers > 0
        :return: cost of the tour
        """
        if len(tour) != self.num_cities or not all(0 <= i < self.cost_mat.shape[0] for i in tour):
            raise ValueError('Tour must be a permutation of cities')

        res = 0
        for i in range(self.num_cities):
            j = (i + 1) % self.num_cities
            res += self.cost(tour[i], tour[j])

        return res

    def __repr__(self):
        """Representation of TSP as a cost matrix"""
        return self.cost_mat.__str__()

    def latex_table(self):
        """Latex cost matrix representation"""

        str_mat = ''
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                str_mat += (str(self.cost_mat[i, j]) if j > i else '') + ' & '
            str_mat += str(i) + '\\\\\n'

        str_mat = str_mat[:-2] + '\n'

        res = ("\\begin{center}\n\\begin{tabular}{ " + ' '.join('c' for _ in range(self.num_cities)) + " | c }\n" +
               ' & '.join(str(i) for i in range(self.num_cities)) + " & \\\\\n" +
               "\\hline\n" + str_mat + "\\end{tabular}\n" + "\\end{center}\n")

        return res


def triu(i, j):
    """Returns the edge in the form (i,j) with j >= i, such that the corresponding matrix is upper triangular. Used in
    the context of the symmetric TSP"""
    if i > j:
        j, i = i, j

    return i, j


if __name__ == '__main__':
    mat = np.triu(np.ones((5, 5)), 1)

    tsp = TSP(mat)

    print(tsp)

    print(tsp.cost(3, 4))

    print(tsp.tour_cost([0, 4, 2, 3, 1]))
