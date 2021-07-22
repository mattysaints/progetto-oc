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

        tsp_cost_mat = np.triu(distance_matrix(coords, coords))

        return TSP(tsp_cost_mat)


def to_mathprog(cost_mat: np.array, path):
    mat_str = ' '.join(str(i + 1) for i in range(cost_mat.shape[1])) + ' :=\n'
    infinite = []
    for i in range(cost_mat.shape[0]):
        mat_str += '\t' + str(i + 1)
        for j in range(cost_mat.shape[1]):
            if cost_mat[i, j] < np.inf:
                mat_str += ' ' + ('.' if j <= i else str(cost_mat[i, j]))
            else:
                infinite.append((i + 1, j + 1))
                mat_str += ' 0'

        if i < cost_mat.shape[0] - 1:
            mat_str += '\n'

    infinite_str = ', '.join(str(e) for e in infinite)

    program = r"""
    set I;
param n := card(I);

set SS := 0 .. (2**n - 1);

set POW {k in SS} := {i in I: (k div 2**(i-1)) mod 2 = 1};

"""+('set INFINITE := {' + infinite_str + '};' if infinite else '')+r"""

set LINKS := {i in I, j in I: i < j};

param cost {LINKS};
var x {LINKS} binary;

minimize TotCost: sum {(i,j) in LINKS} cost[i,j] * x[i,j];

subj to Tour {i in I}: 
   sum {(i,j) in LINKS} x[i,j] + sum {(j,i) in LINKS} x[j,i] = 2;

subj to SubtourElim {k in SS diff {0,2**n-1}}:
   sum {i in POW[k], j in I diff POW[k]: (i,j) in LINKS} x[i,j] +
   sum {i in POW[k], j in I diff POW[k]: (j,i) in LINKS} x[j,i] >= 2;
   
"""+('subj to Inf{(i,j) in INFINITE}: x[i,j] = 0;' if infinite else '')+r"""

solve;

printf "------------------------------------------------------\n";
printf{i in I, j in I: j>i and x[i,j] == 1} "(%d, %d)\n", i-1, j-1;
printf "Cost: %d\n", sum{i in I, j in I: j > i} x[i,j]*cost[i,j];
printf "------------------------------------------------------\n";

data;

set I := """ + (' '.join(str(i + 1) for i in range(cost_mat.shape[0]))) + """;

param cost: """ + mat_str + """;

end;"""

    with open(path, 'w+') as f:
        f.write(program + '\n')


if __name__ == '__main__':
    # inst = parse_tsp('instances/xqf131.tsp')
    #
    # print(inst)

    dist = np.triu([
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
    ]).astype(dtype=np.float64)

    to_mathprog(dist[:5, :5], 'instances/11.tsp')
