import json
import os
import time

import numpy as np

from branchbound import bb_tsp
from bruteforce import bf_tsp
from tsp import TSP
from tsp_parser import to_mathprog


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
    ])[:n, :n])


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


def get_tour(x):
    """Converts a matrix representing a tour to a list of nodes"""
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


def branchbound_13_cities_test():
    """Executes the test of branch and bound algorithm using the 13 cities instance"""
    bb = []
    for n in range(3, 14):
        print(f'N° CITIES: {n}')
        bb.append(cities_13_bb(n))
        print()

    with open('result/bb.json', 'w+') as f:
        json.dump(bb, f)


def bruteforce_13_cities_test():
    """Executes the test of brute-force algorithm using the 13 cities instance, with at most 11 cities"""
    bf = []
    for n in range(3, 12):
        print(f'N° CITIES: {n}')
        bf.append(cities_13_bf(n))
        print()

    with open('result/bf.json', 'w+') as f:
        json.dump(bf, f)


def branchbound_balas_ex1_test():
    """Executes the test of branch and bound algorithm using the graph from example 1 in Balas, Toth [1983]"""
    inst = get_balas_ex1_tsp()

    start = time.time()
    x, z = bb_tsp(inst)
    end = time.time()

    print(f'Best tour: {get_tour(x)}')
    print(f'Cost: {z}')
    print(f'Time: {end - start}')

    print()
    print(inst.to_latex())
    inst.to_graphviz('balas_ex1/balas_ex1.dot')
    inst.to_graphviz('balas_ex1/balas_ex1_tour.dot', tour=x)

    # os.system('neato -Gstart=2 -Tpng balas_ex1/balas_ex1.dot -o balas_ex1/balas_ex1.png -Gdpi=800')
    # os.system('neato -Gstart=2 -Tpng balas_ex1/balas_ex1_tour.dot -o balas_ex1/balas_ex1_tour.png -Gdpi=800')


# def big_test():
#     tsp_instance = parse_tsp('instances/xqf131.tsp')
#
#     start = time.time()
#     best_tour = list(range(tsp_instance.num_cities))
#     bf_tsp(tsp_instance, list(best_tour), best_tour, 0)
#     x, z = bb_tsp(tsp_instance)
#     end = time.time()
#
#     print(x)
#     print(z)
#
#     print(f'Time: {end - start}')

if __name__ == '__main__':
    # bruteforce_13_cities_test()
    # branchbound_13_cities_test()
    branchbound_balas_ex1_test()
