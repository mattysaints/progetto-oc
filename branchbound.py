from collections import Counter

import numpy as np
from tsp import TSP, triu
from kruskal import mst_kruskal


def is_hamiltonian(x):
    for i in range(x.shape[0]):
        count = 0
        for j in range(x.shape[1]):
            if i == j:
                continue

            count += x[triu(i, j)]

        if count != 2:
            return False

    return True


def max_cost_edge(x, included, tsp: TSP):
    for i in range(x.shape[0]):
        count = 0
        max_edge = None
        max_cost = -np.inf

        for j in range(x.shape[1]):
            if i == j:
                continue

            count += x[triu(i, j)]

            if x[triu(i, j)] == 1 and max_cost < tsp.cost(i, j):
                max_edge = triu(i, j)
                max_cost = tsp.cost(i, j)

        if count != 2 and max_edge not in included:
            return max_edge

    return None


def candidates(excluded, included, tsp):
    res = []

    for n in range(tsp.num_cities):

        free_edges = tsp.num_cities - 1
        for ij in excluded:
            if n in ij:
                free_edges -= 1

        num_incl_edges = len([e for e in included if n in e])

        if free_edges >= 2 and num_incl_edges <= 2:
            res.append(n)

    return res


def unfeasible(excluded, tsp: TSP):
    """Returns true if all the edges of a node are excluded, thus the subproblem is unfeasible."""
    num_excl = Counter()

    for i, j in excluded:
        num_excl.update([i, j])

    for count in num_excl.values():
        if count == tsp.num_cities - 1:
            return True

    return False


def bb_tsp(tsp: TSP):
    u = np.inf
    x_p = np.zeros(tsp.cost_mat.shape)
    stack = [(set(), set())]

    while stack:
        excluded, included = stack.pop(0)

        if unfeasible(excluded, tsp):
            continue

        l_nodes = candidates(excluded, included, tsp)

        z_p = np.inf

        for l in l_nodes:
            x_p = mst_kruskal(tsp.cost_mat, l, excluded, included)

            if x_p is None:
                continue

            l_included = list(edge for edge in included if l in edge)
            l_free_edges = (triu(l, i) for i in range(tsp.num_cities) if i != l and triu(l, i) not in included)
            l_edges = sorted(l_free_edges, key=lambda edge: tsp.cost(*edge))[:2 - len(l_included)] + l_included

            for edge in l_edges:
                x_p[edge] = 1

            z_p = np.sum(x_p * tsp.cost_mat)
            break

        if z_p < u:
            if is_hamiltonian(x_p):
                u = z_p
            else:
                edge = max_cost_edge(x_p, included, tsp)

                if not edge:
                    continue

                i, j = edge

                excluded_1 = excluded.union({triu(i, j)})
                included_1 = included

                excluded_2 = excluded
                included_2 = included.union({triu(i, j)})

                stack = [(excluded_1, included_1), (excluded_2, included_2)] + stack

    if u == np.inf:
        return None, np.inf
    else:
        return x_p, u


if __name__ == '__main__':
    tsp = TSP(np.array([
        [0, 1, 2, 8, 1],
        [0, 0, 1, 10, 6],
        [0, 0, 0, 5, 4],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0]
    ]))

    nones = 0
    for _ in range(1000):
        x_star, z_star = bb_tsp(tsp)

        if z_star == np.inf:
            nones += 1

        print(x_star)
        print(z_star)
        print('#########################\n')

    print('###############################\n')
    print(f'Falliti: {nones}')
