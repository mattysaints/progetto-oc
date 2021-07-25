import json
import os
import time

import numpy as np

from branchbound import bb_tsp
from bruteforce import bf_tsp
from tsp import TSP, triu
from tsp_parser import to_mathprog


# INSTANCES ____________________________________________________________________________________________________________

def get_13_cities_tsp(n):
    """Returns a submatrix nxn of the 13 cities instance"""
    return TSP(np.triu([
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]).astype(dtype=np.float64)[:n, :n])


def get_balas_ex1_tsp():
    """Example 1 of Balas, Toth [1983]"""
    return TSP(np.array([
        [0, 2, 4, 5, np.inf, np.inf, np.inf, np.inf],
        [0, 0, 4, np.inf, np.inf, 7, 5, np.inf],
        [0, 0, 0, 1, 7, 4, np.inf, np.inf],
        [0, 0, 0, 0, 10, np.inf, np.inf, np.inf],
        [0, 0, 0, 0, 0, 1, np.inf, 4],
        [0, 0, 0, 0, 0, 0, 3, 5],
        [0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]))


def get_negative_cost_tsp():
    """TSP instance with negative cost edges"""
    return TSP(np.array([
        [0, 10, np.inf, 5, np.inf, np.inf, 14, np.inf, np.inf],
        [0, 0, np.inf, 7, np.inf, 2, np.inf, np.inf, 32],
        [0, 0, 0, 3, np.inf, 12, 1, np.inf, np.inf],
        [0, 0, 0, 0, 10, np.inf, np.inf, -11, np.inf],
        [0, 0, 0, 0, 0, np.inf, 1, np.inf, 9],
        [0, 0, 0, 0, 0, 0, np.inf, -1, -4],
        [0, 0, 0, 0, 0, 0, 0, 1, np.inf],
        [0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]))


# UTILS ________________________________________________________________________________________________________________

def get_tour(x):
    """Converts a matrix representing a tour to a list of nodes"""
    if x is None:
        return None

    coords = []
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[1]):
            if x[i, j] == 1:
                coords.append((i, j))

    tour = list(list((e if e[0] == 0 else e[::-1]) for e in coords if 0 in e)[0])
    node = tour[1]
    while len(tour) < x.shape[0]:
        _, suc = list(
            (e if e[0] == node else e[::-1]) for e in coords if
            node in e and 1 == len(set(e).intersection(set(tour))))[
            0]

        tour.append(suc)
        node = suc

    return tour


def cities_13_glpk(n):
    """13 cities GLPK test"""
    cost_mat = get_13_cities_tsp(n).cost_mat
    to_mathprog(cost_mat, 'instances/cities_13.mod')
    print('======================================')
    os.system('glpsol --math instances/cities_13.mod')


def cities_13_bb(n):
    """13 cities branch and bound test"""
    instance = get_13_cities_tsp(n)

    print('======================================\nBranch and bound:')
    start = time.time()
    x, z = bb_tsp(instance)
    end = time.time()

    print(f'Best tour: {get_tour(x)}')
    print(f'Cost: {z}')
    elapsed = end - start
    print(f'Time: {elapsed}')

    return elapsed


def cities_13_bf(n):
    """13 cities brute force test"""
    instance = get_13_cities_tsp(n)

    print('======================================\nBrute force:')
    start = time.time()
    best_tour = list(range(instance.num_cities))
    bf_tsp(instance, list(best_tour), best_tour, 0)
    end = time.time()

    print(f'Best tour: {best_tour}')
    print(f'Cost: {instance.tour_cost(best_tour)}')
    elapsed = end - start
    print(f'Time: {elapsed}')

    return elapsed


def generate_random_tsp(n, amplitude=100, density=1.0):
    """Generates a random instance with a certain size, density and amplitude"""
    if not (0 <= density <= 1):
        raise ValueError('Density must be in [0,1]')

    cost = np.triu(np.random.rand(n, n), 1) * amplitude
    degree = np.sum(cost < np.inf, axis=0) + np.sum(cost < np.inf, axis=1)
    k = 0
    missing_edges = np.ceil(n * (n - 1) / 2 * (1 - density))
    while k < missing_edges:
        if np.all(degree == 2):
            raise Exception('Unfeasible solution')

        i = np.random.choice([k for k, v in enumerate(degree) if v > 2])
        j = np.random.choice([k for k, v in enumerate(cost[:, i]) if v < np.inf and k < i and degree[k] > 2] +
                             [k for k, v in enumerate(cost[i, :]) if v < np.inf and k > i and degree[k] > 2])

        cost[triu(i, j)] = np.inf
        tmp = cost + np.tril(np.ones((n, n)) * np.nan)
        degree = np.sum(tmp < np.inf, axis=0) + np.sum(tmp < np.inf, axis=1)
        k += 1

    return TSP(cost)


def sparse_15_tsp(n):
    """Random 15 cities instance with amplitude 100 and density 0.5"""
    return TSP(np.array([[0., 65.35407395, 82.02066841, np.inf, 26.42776741,
                          80.22839733, 7.26261536, 20.59938701, 3.45792065, 13.53339106,
                          18.93027924, np.inf, np.inf, np.inf, np.inf],
                         [0., 0., 94.7436637, 35.06230813, 4.60995912,
                          24.65805045, np.inf, 59.30012043, np.inf, np.inf,
                          np.inf, 39.46101065, 61.27724871, 76.77158363, 54.9152151],
                         [0., 0., 0., 51.59221173, 21.43707217,
                          26.39082077, np.inf, 78.02823842, 13.31277999, np.inf,
                          np.inf, np.inf, 32.84121003, np.inf, 3.52892054],
                         [0., 0., 0., 0., np.inf,
                          18.6165595, np.inf, np.inf, np.inf, 0.48668114,
                          17.96264945, 67.81986838, 69.02247542, 65.22721433, 49.71590904],
                         [0., 0., 0., 0., 0.,
                          86.78892068, 39.28652005, 0.45134896, np.inf, 27.26308426,
                          np.inf, 18.38232903, np.inf, 35.15828176, np.inf],
                         [0., 0., 0., 0., 0.,
                          0., np.inf, np.inf, np.inf, 34.92177903,
                          np.inf, 88.17765553, 69.03601435, 23.7098743, 21.54493525],
                         [0., 0., 0., 0., 0.,
                          0., 0., np.inf, np.inf, np.inf,
                          np.inf, np.inf, np.inf, np.inf, 10.06522131],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 24.62976239, 75.12593736,
                          np.inf, 26.69196017, 89.91296827, 83.65234724, 61.76288975],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 14.3781085,
                          np.inf, np.inf, np.inf, np.inf, np.inf],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          np.inf, np.inf, np.inf, np.inf, 83.4599077],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          0., np.inf, np.inf, np.inf, np.inf],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          0., 0., np.inf, np.inf, np.inf],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          0., 0., 0., np.inf, np.inf],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          0., 0., 0., 0., np.inf],
                         [0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.]])[:n, :n])


def sparse_13_bf(n):
    """13 cities sparse brute-force test"""
    instance = sparse_15_tsp(n)

    print('======================================\nBrute-force:')
    start = time.time()
    best_tour = list(range(instance.num_cities))
    bf_tsp(instance, list(best_tour), best_tour, 0)
    end = time.time()

    print(f'Best tour: {best_tour}')
    print(f'Cost: {instance.tour_cost(best_tour)}')
    elapsed = end - start
    print(f'Time: {elapsed}')

    return elapsed


def sparse_13_bb(n):
    """13 cities sparse branch and bound test"""
    instance = sparse_15_tsp(n)

    print('======================================\nBranch and bound:')
    start = time.time()
    x, z = bb_tsp(instance)
    end = time.time()

    print(f'Best tour: {get_tour(x)}')
    print(f'Cost: {z}')
    elapsed = end - start
    print(f'Time: {elapsed}')

    return elapsed


# TESTS ________________________________________________________________________________________________________________

def time_13_cities_to_json():
    """Saves in a JSON file the time of execution of the BB and BF algorithms on the 13 cities instance"""
    bb = []
    for n in range(3, 14):
        print(f'N° CITIES: {n}')
        bb.append(cities_13_bb(n))
        print()

    with open('versus/bb.json', 'w+') as f:
        json.dump(bb, f)

    bf = []
    for n in range(3, 12):
        print(f'N° CITIES: {n}')
        bf.append(cities_13_bf(n))
        print()

    with open('versus/bf.json', 'w+') as f:
        json.dump(bf, f)


def branchbound_13_cities_test(glpk=False):
    """Executes the test of branch and bound algorithm using the 13 cities instance: 170 secs circa"""
    if glpk:
        cities_13_glpk(13)

    cities_13_bb(13)


def bruteforce_11_cities_test(glpk=False):
    """Executes the test of brute-force algorithm using the 13 cities instance with 11 cities: 350 secs circa"""
    if glpk:
        cities_13_glpk(13)

    cities_13_bf(13)


def branchbound_balas_ex1_test(glpk=False, save=False):
    """Executes the test of branch and bound algorithm using the graph from example 1 in Balas, Toth [1983]"""
    inst = get_balas_ex1_tsp()

    if glpk:
        to_mathprog(inst.cost_mat, 'instances/balas_ex1.mod')
        os.system('glpsol --math instances/balas_ex1.mod')

    print('======================================\nBranch and bound:')
    start = time.time()
    x, z = bb_tsp(inst)
    end = time.time()

    print(f'Best tour: {get_tour(x)}')
    print(f'Cost: {z}')
    print(f'Time: {end - start}')

    if save:
        print()
        print(inst.to_latex())
        inst.to_graphviz('balas_ex1/balas_ex1.dot')
        inst.to_graphviz('balas_ex1/balas_ex1_tour.dot', tour=x)

        os.system('neato -Gstart=2 -Tpng balas_ex1/balas_ex1.dot -o balas_ex1/balas_ex1.png -Gdpi=800')
        os.system('neato -Gstart=2 -Tpng balas_ex1/balas_ex1_tour.dot -o balas_ex1/balas_ex1_tour.png -Gdpi=800')


def branchbound_negative_cost_test(glpk=False, save=False):
    """Executes the test of branch and bound algorithm using a graph with negative cost edges"""
    inst = get_negative_cost_tsp()

    if glpk:
        to_mathprog(inst.cost_mat, 'instances/negative_cost.mod')
        os.system('glpsol --math instances/negative_cost.mod')

    start = time.time()
    x, z = bb_tsp(inst)
    end = time.time()

    print('======================================\nBranch and bound:')
    print(f'Best tour: {get_tour(x)}')
    print(f'Cost: {z}')
    print(f'Time: {end - start}')

    if save:
        print()
        print(inst.to_latex())

        inst.to_graphviz('negative_cost/instance.dot', tour=x)

        os.system('neato -Gstart=5 -Tpng negative_cost/instance.dot -o negative_cost/instance.png -Gdpi=800')


def time_13_sparse_to_json():
    """Saves in a JSON file the time of execution of the BB algorithm on the 13 cities sparse instance. The brute-force
    is not executed because the sparseness of the graph doesn't change it's execution time"""
    bb = []
    for n in range(3, 14):
        print(f'N° CITIES: {n}')
        bb.append(sparse_13_bb(n))
        print()

    with open('versus/bb_sparse.json', 'w+') as f:
        json.dump(bb, f)


def branchbound_15_sparse_test(glpk=False, save=False):
    """Executes the test of branch and bound algorithm using a random graph with 15 nodes"""
    inst = sparse_15_tsp(15)

    if glpk:
        to_mathprog(inst.cost_mat, 'instances/random.mod')
        os.system('glpsol --math instances/random.mod')

    print('======================================\nBranch and bound:')
    start = time.time()
    x, z = bb_tsp(inst)
    end = time.time()

    print(f'Best tour: {get_tour(x)}')
    print(f'Cost: {z}')
    print(f'Time: {end - start}')

    if save:
        print(inst.to_latex())


def branchbound_random(n, amplitude=100, density=0.5, glpk=False, save=False):
    """Executes the test of branch and bound algorithm using a random graph"""
    inst = generate_random_tsp(n, amplitude, density)

    if glpk:
        to_mathprog(inst.cost_mat, 'instances/random.mod')
        os.system('glpsol --math instances/random.mod')

    print('======================================\nBranch and bound:')
    start = time.time()
    x, z = bb_tsp(inst)
    end = time.time()

    print(f'Best tour: {get_tour(x)}')
    print(f'Cost: {z}')
    print(f'Time: {end - start}')

    if save:
        print(inst.to_latex())


if __name__ == '__main__':
    # branchbound_balas_ex1_test(glpk=False)
    branchbound_13_cities_test(glpk=False)
    # branchbound_random(16,density=.5,glpk=False)