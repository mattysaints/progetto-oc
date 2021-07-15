import numpy as np
from scipy.spatial import distance_matrix

from tsp import TSP


def parse_tsp(path):
    with open(path, 'r') as f:
        coords = []
        lines = [l.strip() for l in f.readlines() if l.strip() != '' and l[0].isdigit()]

        for l in lines:
            _, i, j = l.split(' ')
            coords.append([int(i), int(j)])

        tsp_cost_mat = np.triu(distance_matrix(coords,coords))

        return TSP(tsp_cost_mat)


if __name__ == '__main__':
    inst = parse_tsp('instances/xqf131.tsp')

    print(inst)