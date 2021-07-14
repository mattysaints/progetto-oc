import numpy as np
from tsp import TSP
from kruskal import mst_kruskal


def is_hamiltonian(x):
    n = x.shape[0]

    for i in range(n):
        count = 0
        for j in range(n):
            ip, jp = (j, i) if j < i else (i, j)
            count += x[ip, jp]

        if count != 2:
            return False

    return True


def max_cost_edge(x, excluded, included, tsp: TSP):
    n = x.shape[0]

    for i in range(n):
        count = 0
        max_edge = (0, 0)
        max_cost = 0

        for j in range(n):
            ip, jp = (j, i) if j <= i else (i, j)
            count += x[ip, jp]

            if max_cost < x[ip, jp] * tsp.cost(ip, jp):
                max_edge = (ip, jp)
                max_cost = tsp.cost(ip, jp)

        if count != 2 and max_edge not in included and max_edge not in excluded:
            return max_edge

    raise Exception('Hamiltonian cycle')


def choose_node(excluded, tsp):
    candidates = []

    for n in range(tsp.num_cities):

        free_edges = tsp.num_cities - 1
        for ij in excluded:
            if n in ij:
                free_edges -= 1

        if free_edges >= 2:
            candidates.append(n)

    return np.random.choice(candidates)


def bb_tsp(tsp: TSP):
    u = np.inf
    x_p = np.zeros(tsp.cost_mat.shape)
    stack = [(set(), set())]

    node = 0

    while stack:
        print(node)
        node += 1

        excluded, included = stack.pop(0)
        # print(f'Esclusi: {excluded}')
        # print(f'Inclusi: {included}\n')
        # input('>')

        l = choose_node(excluded, tsp)

        x_p, mst_cost = mst_kruskal(tsp.cost_mat, l, excluded, included)

        l_included = list(edge for edge in included if l in edge)
        if len(l_included) > 2:
            print('Più di 2 inclusi')

        l_edges = sorted(((l, i) for i in range(tsp.num_cities) if i != l), key=lambda edge: tsp.cost(*edge))[:2-len(l_included)] + l_included

        l_edges = list((edge[::-1] if edge[0] > edge[1] else edge) for edge in l_edges)

        z_p = mst_cost

        for edge in l_edges:
            x_p[edge] = 1
            z_p += tsp.cost(*edge)

        if z_p < u:
            if is_hamiltonian(x_p):
                u = z_p
            else:
                i, j = max_cost_edge(x_p, excluded, included, tsp)

                excluded_1 = excluded.union({(i, j)})
                included_1 = included

                excluded_2 = excluded
                included_2 = included.union({(i, j)})

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

    for _ in range(100):
        x, z = bb_tsp(tsp)

        print(x)
        print(z)
