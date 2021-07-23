import pprint
from collections import Counter

import numpy as np

from kruskal import mst_kruskal
from tsp import TSP, triu


def is_hamiltonian(x):
    """Returns true if the graph (boolean matrix) is a Hamiltonian Cycle"""
    count = np.sum(x, axis=0) + np.sum(x, axis=1)
    return np.all(count == 2)


def max_cost_edge(x, included, tsp: TSP):
    """Returns the maximum cost edge of a node with degree d != 2, if such edge is not included. Returns None if
    such edge doesn't exists"""
    for i in range(x.shape[0]):
        count = 0
        max_edge = None
        max_cost = -np.inf

        for j in range(x.shape[1]):
            if i == j:
                continue

            count += x[triu(i, j)]

            if x[triu(i, j)] == 1 and max_cost < tsp.cost(i, j) and triu(i, j) not in included:
                max_edge = triu(i, j)
                max_cost = tsp.cost(i, j)

        if count != 2 and max_edge is not None:
            return max_edge

    return None


def unfeasible(excluded, included, tsp: TSP):
    """Returns true if all the edges of a node are excluded, thus the subproblem is unfeasible."""
    for n in range(tsp.num_cities):

        free_edges = tsp.num_cities - 1
        for ij in excluded:
            if n in ij:
                free_edges -= 1

        num_incl_edges = len([e for e in included if n in e])

        if free_edges < 2 or num_incl_edges > 2:
            return True

    return False


def min_cost_1_tree(l, excluded, included, tsp):
    """Returns the minimum cost 1-tree with respect to fixed edges. Returns None if doesn't exists"""
    x_p = mst_kruskal(tsp.cost_mat, l, excluded, included)

    if x_p is not None:
        l_included = list(edge for edge in included if l in edge)
        l_free_edges = (triu(l, i) for i in range(tsp.num_cities)
                        if i != l and triu(l, i) not in excluded and triu(l, i) not in included)
        l_edges = sorted(l_free_edges, key=lambda edge: tsp.cost(*edge))[:2 - len(l_included)] + l_included

        for edge in l_edges:
            x_p[edge] = 1

        return x_p, np.nansum(x_p * tsp.cost_mat)
    else:
        return None, np.inf


def bb_tsp(tsp: TSP):
    """
    Branch and bound algorithm for the symmetric TSP based on the 1-tree relaxation.

    :param tsp: TSP instance
    :return: optimal solution and the optimal value, if the problem has a solution
    """
    u = np.inf
    best_tour = None
    x_p = np.zeros(tsp.cost_mat.shape)
    stack = [(set(), set())]

    while stack:
        excluded, included = stack.pop(0)

        if not unfeasible(excluded, included, tsp):
            z_p = np.inf

            for l in range(tsp.num_cities):
                x_p, z_p = min_cost_1_tree(l, excluded, included, tsp)

                if x_p is not None:
                    break

            if z_p < u:
                if is_hamiltonian(x_p):
                    u = z_p
                    best_tour = np.array(x_p)
                else:
                    edge = max_cost_edge(x_p, included, tsp)

                    if edge:
                        i, j = edge
                        stack = [(excluded | {triu(i, j)}, included), (excluded, included | {triu(i, j)})] + stack

    if u == np.inf:
        return None, np.inf
    else:
        return best_tour, u


if __name__ == '__main__':
    tsp = TSP(np.array([
        [0, 1, 2, 8, 1],
        [0, 0, 1, 10, 6],
        [0, 0, 0, 5, 4],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0]
    ]))

    x, z = bb_tsp(tsp)

    print(x)
    print(z)
