import numpy as np
from tsp import TSP


def bf_tsp(tsp: TSP, tour: np.array, best_tour: np.array, j):
    """
    Bruteforce solution for TSP. It generates all possible permutations and keeps the best tour.

    :param tsp: TSP instance
    :param tour: partial tour
    :param best_tour: best tour so far
    :param j: indicates that tour[0..j-1] is fixed
    """
    if j == tsp.num_cities:
        if tsp.tour_cost(tour) < tsp.tour_cost(best_tour):
            best_tour[:] = tour[:]
    else:
        for i in range(j, len(tour)):
            tour[i], tour[j] = tour[j], tour[i]
            bf_tsp(tsp, tour, best_tour, j + 1)
            tour[i], tour[j] = tour[j], tour[i]


if __name__ == '__main__':
    tsp = TSP(np.array([
        [0, 25, 25, 13],
        [0, 0, 25, 13],
        [0, 0, 0, 13],
        [0, 0, 0, 0]
    ]))

    tour, best_tour = np.array(range(4)), np.array(range(4))

    bf_tsp(tsp, tour, best_tour, 0)

    print(best_tour)
    print(tsp.tour_cost(best_tour))
